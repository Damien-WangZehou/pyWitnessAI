/* Bundled into the two Qualtrics questions by build_qsf.py. */
function createLineupClient(engine) {
    'use strict';
    var active = true;
    var timer = null;
    var releaseDelay = null;
    var controller = null;
    var config = {
        participant_id: String(engine.getEmbeddedData('lineupSeed') || ''),
        description: String(engine.getEmbeddedData('verbalDescription') || '').trim(),
        condition: String(engine.getEmbeddedData('condition') || ''),
        target_lineup: String(engine.getEmbeddedData('targetLineup') || ''),
        study_version: String(engine.getEmbeddedData('studyVersion') || '')
    };
    var base = String(engine.getEmbeddedData('LINEUP_API_BASE_URL') || '').trim().replace(/\/+$/, '');

    function set(name, value) {
        if (active) engine.setEmbeddedData(name, value == null ? '' : String(value));
    }

    function validate(data) {
        if (!data || data.condition !== config.condition || data.target_lineup !== config.target_lineup ||
            data.study_version !== config.study_version || !data.lineup_id) {
            throw new Error('The generation service returned a mismatched study assignment.');
        }
        if (['queued', 'generating', 'ready', 'error'].indexOf(data.status) < 0) {
            throw new Error('Invalid generation status.');
        }
        if (data.status === 'ready') {
            if (!Array.isArray(data.items) || data.items.length !== 6) throw new Error('Six lineup members are required.');
            var ids = {};
            var urls = {};
            var suspects = 0;
            data.items.forEach(function(item, index) {
                if (!item || !item.image_id || ids[item.image_id] || Number(item.position) !== index + 1 ||
                    ['suspect', 'filler'].indexOf(item.role) < 0 || !item.image_url) {
                    throw new Error('Invalid lineup member.');
                }
                ids[item.image_id] = true;
                if (item.role === 'suspect') suspects += 1;
                item.image_url = new URL(item.image_url, base + '/').href;
                if (!/^https:\/\//i.test(item.image_url)) throw new Error('Lineup media must use HTTPS.');
                if (urls[item.image_url]) throw new Error('Lineup media must be unique.');
                urls[item.image_url] = true;
            });
            if (suspects !== (config.target_lineup === 'targetPresent' ? 1 : 0)) {
                throw new Error('The lineup does not match the target assignment.');
            }
        }
        return data;
    }

    function persist(data) {
        validate(data);
        set('generationStatus', data.status);
        set('lineupID', data.lineup_id);
        if (data.status === 'ready') {
            set('generationError', '');
            set('generationMetadata', JSON.stringify(data.metadata || {}));
            set('suspectPosition', '');
            data.items.forEach(function(item, index) {
                var prefix = 'lineup' + (index + 1);
                set(prefix, item.image_id);
                set(prefix + 'Url', item.image_url);
                set(prefix + 'Role', item.role);
                if (item.role === 'suspect') set('suspectPosition', index + 1);
            });
        }
        return data;
    }

    function request(path, body) {
        if (!active) return Promise.reject(new Error('Page closed.'));
        controller = new AbortController();
        var timeout = setTimeout(function() { controller.abort(); }, 25000);
        var options = {method: body ? 'POST' : 'GET', mode: 'cors', credentials: 'omit',
            referrerPolicy: 'no-referrer', cache: 'no-store', signal: controller.signal};
        if (body) {
            options.headers = {'Content-Type': 'application/json'};
            options.body = JSON.stringify(body);
        }
        return fetch(base + path, options).then(function(response) {
            if (!response.ok) throw new Error('Generation service HTTP ' + response.status + '.');
            return response.json();
        }).then(persist).then(function(data) {
            clearTimeout(timeout);
            return data;
        }, function(error) {
            clearTimeout(timeout);
            throw error;
        });
    }

    function start() {
        if (!/^https:\/\//i.test(base) || /YOUR-|example\./i.test(base)) {
            return Promise.reject(new Error('LINEUP_API_BASE_URL is not configured.'));
        }
        if (!config.participant_id || !config.description || config.description.length > 2000 ||
            ['mixingFaces', 'noMixing'].indexOf(config.condition) < 0 ||
            ['targetPresent', 'targetAbsent'].indexOf(config.target_lineup) < 0 ||
            config.study_version !== 'ai_generated_lineup_2x2_v1') {
            return Promise.reject(new Error('Missing or invalid study assignment or description.'));
        }
        set('lineupPreloadStarted', 'true');
        return request('/v1/lineups', config);
    }

    function wait(onProgress) {
        var started = Date.now();
        function poll(data) {
            if (!active) throw new Error('Page closed.');
            if (data.status === 'error') throw new Error('Generation failed; contact the study researcher.');
            if (data.status === 'ready') return data;
            if (Date.now() - started > 900000) throw new Error('Lineup preparation timed out.');
            if (onProgress) onProgress(data.status);
            return new Promise(function(resolve) {
                releaseDelay = resolve;
                timer = setTimeout(resolve, 3000);
            }).then(function() {
                if (!active) throw new Error('Page closed.');
                return request('/v1/lineups/' + encodeURIComponent(data.lineup_id));
            }).then(poll);
        }
        return start().then(poll);
    }

    return {start: start, wait: wait, validate: validate, set: set, config: config,
        cancel: function() {
            active = false;
            clearTimeout(timer);
            if (releaseDelay) releaseDelay();
            if (controller) controller.abort();
        }};
}
