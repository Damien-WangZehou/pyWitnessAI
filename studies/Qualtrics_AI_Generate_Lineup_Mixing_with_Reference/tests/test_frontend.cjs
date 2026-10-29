/* Exercise the shipped Qualtrics scripts with a minimal DOM and mocked HTTP.
 * No browser account, network request, or paid image generation is used.
 * Run: node --test studies/.../tests/test_frontend.cjs
 */
'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const folder = path.resolve(__dirname, '..');
const clientSource = fs.readFileSync(path.join(folder, 'lineup_client.js'), 'utf8');
const questionSource = fs.readFileSync(path.join(folder, 'lineup_question.js'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
const flush = async () => { for (let i = 0; i < 25; i += 1) await Promise.resolve(); };

class Element {
    constructor(tag = 'div') {
        this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
        this.style = {}; this.attributes = {}; this.events = {}; this.hidden = false;
    }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() { return this.text || ''; }
    appendChild(child) { this.children.push(child); return child; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name]; }
    addEventListener(name, handler) { (this.events[name] ||= []).push(handler); }
    click() { for (const handler of this.events.click || []) handler({preventDefault() {}}); }
    querySelectorAll(selector) {
        if (selector === '[data-choice]') return this.children.filter(c => c.dataset.choice);
        return [];
    }
}

function data(condition = 'noMixing', target = 'targetAbsent', status = 'ready') {
    return {status, lineup_id: 'stable_job', study_version: 'ai_generated_lineup_2x2_v1',
        condition, target_lineup: target, metadata: {prompt_processing: false},
        items: Array.from({length: 6}, (_, i) => ({position: i + 1,
            image_id: `face_${i + 1}`, image_url: `/media/stable_job/${i + 1}.png`,
            role: i === 2 && target === 'targetPresent' ? 'suspect' : 'filler'}))};
}

function fixture({condition = 'noMixing', target = 'targetAbsent', fetchResponse, question = false} = {}) {
    const fields = {lineupSeed: 'response-random123', verbalDescription: 'A MAN with Short Hair.',
        condition, targetLineup: target, studyVersion: 'ai_generated_lineup_2x2_v1',
        LINEUP_API_BASE_URL: 'https://lineup.study.test'};
    const callbacks = {load: [], ready: [], submit: [], unload: []};
    const engine = {getEmbeddedData: key => fields[key], setEmbeddedData: (key, value) => { fields[key] = value; }};
    for (const [name, key] of [['addOnload', 'load'], ['addOnReady', 'ready'], ['addOnPageSubmit', 'submit'], ['addOnUnload', 'unload']]) {
        engine[name] = fn => callbacks[key].push(fn);
    }
    const nodes = Object.fromEntries(['.ChoiceStructure', '[data-lineup-status]', '[data-lineup-grid]', '[data-lineup-retry]', '[data-lineup-exit]'].map(s => [s, new Element()]));
    const container = {querySelector: selector => nodes[selector] || null};
    const calls = [], images = [], timers = new Map();
    let time = 0, serial = 0;
    class ImageElement extends Element {
        constructor() { super('img'); images.push(this); }
        finish() { if (this.onload) this.onload(); }
    }
    const sandbox = {URL, AbortController, Image: ImageElement, Promise, Date,
        document: {createElement: tag => new Element(tag)}, performance: {now: () => time},
        Qualtrics: {SurveyEngine: engine},
        setTimeout: (fn, delay) => { const id = ++serial; timers.set(id, {fn, delay}); return id; },
        clearTimeout: id => timers.delete(id),
        fetch: async (url, options) => {
            calls.push({url, options});
            const payload = fetchResponse ? await fetchResponse(url, options, calls.length) : data(condition, target);
            return {ok: true, status: 200, json: async () => clone(payload)};
        }};
    vm.createContext(sandbox);
    vm.runInContext(clientSource, sandbox, {filename: 'lineup_client.js'});
    const nativeValues = new Map();
    const q = {next: false, hideNextButton() { this.next = false; }, showNextButton() { this.next = true; },
        getQuestionContainer: () => container,
        setChoiceValue(choice, value) {
            if (value) nativeValues.clear();
            nativeValues.set(choice, value);
        }};
    if (question) {
        vm.runInContext(questionSource, sandbox, {filename: 'lineup_question.js'});
        for (const fn of callbacks.load) fn.call(q);
        for (const fn of callbacks.ready) fn.call(q);
    }
    return {fields, nodes, callbacks, q, images, calls, nativeValues,
        client: sandbox.createLineupClient(engine),
        tick: value => { time = value; },
        firePoll: () => {
            const timer = [...timers].find(([, t]) => t.delay === 3000);
            assert.ok(timer, 'a polling timer exists'); timers.delete(timer[0]); timer[1].fn();
        }};
}

for (const condition of ['mixingFaces', 'noMixing']) {
    for (const target of ['targetPresent', 'targetAbsent']) {
        test(`contract and data persistence: ${condition} / ${target}`, async () => {
            const f = fixture({condition, target});
            const result = await f.client.start();
            assert.equal(result.items.length, 6);
            assert.equal(f.fields.condition, condition);
            assert.equal(f.fields.targetLineup, target);
            assert.equal(f.fields.suspectPosition, target === 'targetPresent' ? '3' : '');
            assert.equal(f.fields.lineup3Role, target === 'targetPresent' ? 'suspect' : 'filler');
            assert.equal(f.fields.lineup1Url, 'https://lineup.study.test/media/stable_job/1.png');
            const body = JSON.parse(f.calls[0].options.body);
            assert.equal(body.description, 'A MAN with Short Hair.');
            assert.equal(body.condition, condition); assert.equal(body.target_lineup, target);
        });

        test(`identification, rejection, native choice and RT: ${condition} / ${target}`, async () => {
            const f = fixture({condition, target, question: true});
            const grid = f.nodes['[data-lineup-grid]'];
            assert.equal(f.q.next, false);
            await flush();
            assert.equal(f.images.length, 6);
            f.tick(9000);
            f.images.slice(0, 5).forEach(img => img.finish());
            await flush();
            assert.equal(grid.hidden, true, 'partial lineup is never displayed');
            assert.equal(f.q.next, false);
            f.tick(10000); f.images[5].finish(); await flush();
            assert.equal(grid.hidden, false);
            assert.equal(grid.children.length, 7);
            assert.equal(f.q.next, false, 'next requires an identification choice');
            f.tick(11250); grid.children[2].click();
            assert.equal(f.fields.responseType, target === 'targetPresent' ? 'suspectId' : 'fillerId');
            assert.equal(f.fields.SelectedPosition, '3');
            assert.equal(f.fields.SelectedImage, 'face_3');
            assert.equal(f.fields.LineupDecisionRTms, '1250', 'RT starts after last image loads');
            assert.equal(f.fields.FirstResponseType, target === 'targetPresent' ? 'suspectId' : 'fillerId');
            assert.equal(f.fields.FirstSelectedPosition, '3');
            assert.equal(f.fields.FirstSelectedImage, 'face_3');
            assert.equal(f.q.next, true); assert.equal(f.nativeValues.get(3), true);
            f.tick(12000); grid.children[0].click();
            assert.equal(f.fields.responseType, 'fillerId');
            f.tick(13000); grid.children[6].click();
            assert.equal(f.fields.responseType, 'rejectId');
            assert.equal(f.fields.FirstSelectedPosition, '3', 'first choice remains available after revisions');
            assert.equal(f.fields.Outcome, target === 'targetPresent' ? 'False Rejection' : 'Correct Rejection');
            assert.equal(f.fields.LineupDecisionRTms, '1250', 'first-response RT stays fixed');
            for (const fn of f.callbacks.submit) fn.call(f.q, 'next');
            assert.equal(f.fields.LineupFinalDecisionRTms, '3000');
            assert.equal(f.nativeValues.get(7), true);
            for (const fn of f.callbacks.unload) fn.call(f.q);
        });
    }
}

test('reject mismatched assignments, roles, duplicate IDs, positions and insecure media', () => {
    const f = fixture();
    for (const mutate of [
        x => { x.condition = 'mixingFaces'; },
        x => { x.target_lineup = 'targetPresent'; },
        x => { x.study_version = 'obsolete-study-version'; },
        x => { x.items[0].role = 'suspect'; },
        x => { x.items[0].role = 'designated'; },
        x => { x.items[1].image_id = x.items[0].image_id; },
        x => { x.items[1].image_url = x.items[0].image_url; },
        x => { x.items[1].position = 3; },
        x => { x.items[1].image_url = 'http://insecure.test/x.png'; },
        x => { x.items.pop(); },
        x => { x.status = 'unknown'; },
    ]) {
        const payload = data(); mutate(payload); assert.throws(() => f.client.validate(payload));
    }
    const tp = fixture({target: 'targetPresent'});
    const missing = data('noMixing', 'targetPresent'); missing.items[2].role = 'filler';
    assert.throws(() => tp.client.validate(missing));
});

test('queued jobs poll with GET; repeated start preserves the same idempotency payload', async () => {
    let getCount = 0;
    const f = fixture({fetchResponse: async (_url, options) => {
        if (options.method === 'POST') return data('noMixing', 'targetAbsent', 'queued');
        getCount += 1;
        return data('noMixing', 'targetAbsent', getCount === 1 ? 'generating' : 'ready');
    }});
    const pending = f.client.wait(); await flush();
    f.firePoll(); await flush(); f.firePoll();
    const result = await pending;
    assert.equal(result.status, 'ready');
    assert.deepEqual(f.calls.map(c => c.options.method), ['POST', 'GET', 'GET']);
    assert.ok(f.calls.slice(1).every(c => c.url.endsWith('/v1/lineups/stable_job')));
    await f.client.start();
    assert.equal(f.calls[0].options.body, f.calls[3].options.body, 'retry cannot request a new assignment or seed');
    f.client.cancel();
});

test('technical exit is distinct from identification and creates no fabricated confidence', async () => {
    const f = fixture({question: true, fetchResponse: async () => { throw new Error('Offline'); }});
    await flush();
    assert.equal(f.q.next, false);
    assert.equal(f.nodes['[data-lineup-exit]'].hidden, false);
    assert.equal(f.nodes['[data-lineup-retry]'].hidden, false);
    f.nodes['[data-lineup-exit]'].click();
    assert.equal(f.fields.technicalFailure, 'true');
    assert.equal(f.fields.SelectionMade, 'false');
    assert.equal(f.fields.responseType, '');
    assert.equal(f.fields.confidence, undefined);
    assert.equal(f.fields.LineupDecisionRTms, '');
    assert.equal(f.nativeValues.get(8), true);
    assert.equal(f.q.next, true);
    for (const fn of f.callbacks.unload) fn.call(f.q);
});

test('cancel stops pending polling without writing data on the next page', async () => {
    const f = fixture({fetchResponse: async () => data('noMixing', 'targetAbsent', 'queued')});
    const pending = f.client.wait(); await flush();
    const before = clone(f.fields);
    f.client.cancel();
    await assert.rejects(pending, /Page closed/);
    assert.deepEqual(f.fields, before);
    assert.equal(f.calls.length, 1);
});
