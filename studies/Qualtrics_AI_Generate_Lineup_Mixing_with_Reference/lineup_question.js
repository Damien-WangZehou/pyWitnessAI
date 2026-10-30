Qualtrics.SurveyEngine.addOnload(function() {
    this.hideNextButton();
});

Qualtrics.SurveyEngine.addOnReady(function() {
    'use strict';
    var question = this;
    var client = createLineupClient(Qualtrics.SurveyEngine);
    var container = this.getQuestionContainer();
    var nativeChoices = container.querySelector('.ChoiceStructure');
    var status = container.querySelector('[data-lineup-status]');
    var grid = container.querySelector('[data-lineup-grid]');
    var retry = container.querySelector('[data-lineup-retry]');
    var technicalExit = container.querySelector('[data-lineup-exit]');
    var items = [];
    var readyAt = null;
    var firstChoiceAt = null;
    var pageOpenedAt = performance.now();
    var selection = null;
    var alive = true;
    var busy = false;
    var attempts = Number(Qualtrics.SurveyEngine.getEmbeddedData('technicalFailureAttempts') || 0);
    if (nativeChoices) nativeChoices.style.display = 'none';

    function field(name, value) { client.set(name, value); }
    function resetDecision() {
        selection = null;
        firstChoiceAt = null;
        question.hideNextButton();
        for (var n = 1; n <= 8; n += 1) question.setChoiceValue(n, false);
        ['responseType', 'SelectedPosition', 'SelectedImage', 'Outcome', 'LineupDecisionRTms',
            'LineupDecisionRTs', 'LineupFinalDecisionRTms', 'FirstResponseType', 'FirstSelectedPosition',
            'FirstSelectedImage'].forEach(function(name) { field(name, ''); });
        field('SelectionMade', 'false');
        field('technicalFailure', 'false');
        field('technicalFailureReason', '');
    }

    function record(choice) {
        if (choice !== 8 && readyAt === null) return;
        selection = choice;
        question.setChoiceValue(choice, true);
        var type = '';
        if (choice <= 6) {
            var member = items[choice - 1];
            type = member.role === 'suspect' ? 'suspectId' : 'fillerId';
            field('SelectedPosition', choice);
            field('SelectedImage', member.image_id);
            field('Outcome', type === 'suspectId' ? 'Correct Identification' : 'Filler Identification');
        } else if (choice === 7) {
            type = 'rejectId';
            field('SelectedPosition', 'NotPresent');
            field('SelectedImage', '');
            field('Outcome', client.config.target_lineup === 'targetPresent' ? 'False Rejection' : 'Correct Rejection');
        } else {
            field('SelectedPosition', 'TechnicalFailure');
            field('SelectedImage', '');
            field('Outcome', 'Technical Failure');
            field('technicalFailure', 'true');
        }
        field('responseType', type);
        field('SelectionMade', choice === 8 ? 'false' : 'true');
        if (choice <= 7 && firstChoiceAt === null) {
            firstChoiceAt = performance.now();
            var elapsed = Math.max(0, Math.round(firstChoiceAt - readyAt));
            field('LineupDecisionRTms', elapsed);
            field('LineupDecisionRTs', (elapsed / 1000).toFixed(3));
            field('FirstResponseType', type);
            field('FirstSelectedPosition', choice === 7 ? 'NotPresent' : choice);
            field('FirstSelectedImage', choice <= 6 ? items[choice - 1].image_id : '');
        }
        Array.prototype.forEach.call(grid.querySelectorAll('[data-choice]'), function(button) {
            button.setAttribute('aria-checked', String(Number(button.dataset.choice) === choice));
        });
        question.showNextButton();
    }

    function buttonFor(choice, text, image) {
        var button = document.createElement('button');
        button.type = 'button';
        button.className = image ? 'ai-lineup-member' : 'ai-lineup-reject';
        button.dataset.choice = String(choice);
        button.setAttribute('role', 'radio');
        button.setAttribute('aria-checked', 'false');
        if (image) button.appendChild(image);
        var label = document.createElement('span');
        label.textContent = text;
        button.appendChild(label);
        button.addEventListener('click', function() { record(choice); });
        return button;
    }

    function show(data) {
        var loads = data.items.map(function(item, index) {
            return new Promise(function(resolve, reject) {
                var img = new Image();
                var timeout = setTimeout(function() { reject(new Error('Lineup image timed out.')); }, 30000);
                img.alt = 'Lineup member ' + (index + 1);
                img.referrerPolicy = 'no-referrer';
                img.onload = function() { clearTimeout(timeout); resolve(img); };
                img.onerror = function() { clearTimeout(timeout); reject(new Error('Lineup image could not be loaded.')); };
                img.src = item.image_url;
            });
        });
        return Promise.all(loads).then(function(images) {
            if (!alive) return;
            items = data.items;
            grid.textContent = '';
            images.forEach(function(img, index) { grid.appendChild(buttonFor(index + 1, 'Person ' + (index + 1), img)); });
            grid.appendChild(buttonFor(7, 'Not Present'));
            grid.hidden = false;
            readyAt = performance.now();
            field('lineupLoadTimeMs', Math.round(readyAt - pageOpenedAt));
            field('generationStatus', 'ready');
            field('generationError', '');
            status.textContent = 'Select one face, or choose Not Present.';
        });
    }

    function fail(error) {
        if (!alive) return;
        attempts += 1;
        field('technicalFailureAttempts', attempts);
        field('generationError', String(error.message || error).slice(0, 500));
        field('technicalFailureReason', String(error.message || error).slice(0, 500));
        field('generationStatus', 'error');
        status.textContent = 'We could not prepare the lineup. You can retry, or end the study and receive the completion code.';
        retry.hidden = false;
        technicalExit.hidden = false;
    }

    function run() {
        if (busy) return;
        busy = true;
        resetDecision();
        readyAt = null;
        grid.hidden = true;
        retry.hidden = true;
        technicalExit.hidden = true;
        status.textContent = 'Preparing your lineup. Please wait; this may take several minutes.';
        client.wait(function() {
            status.textContent = 'Preparing your lineup. Please keep this page open.';
        }).then(show).catch(fail).then(function() { busy = false; });
    }
    retry.addEventListener('click', run);
    grid.addEventListener('keydown', function(event) {
        var buttons = Array.prototype.slice.call(grid.querySelectorAll('[data-choice]'));
        var index = buttons.indexOf(document.activeElement);
        if (index < 0) return;
        var delta = {ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1}[event.key];
        if (!delta && event.key !== 'Home' && event.key !== 'End') return;
        event.preventDefault();
        var next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
            (index + delta + buttons.length) % buttons.length;
        buttons[next].focus();
        buttons[next].click();
    });
    technicalExit.addEventListener('click', function() {
        record(8);
        retry.hidden = true;
        technicalExit.hidden = true;
        status.textContent = 'Select the next arrow to receive your completion code.';
    });
    question._aiSubmit = function() {
        if (selection !== null) {
            record(selection);
            if (selection <= 7) field('LineupFinalDecisionRTms', Math.max(0, Math.round(performance.now() - readyAt)));
        }
    };
    question._aiCleanup = function() { alive = false; client.cancel(); };
    run();
});

Qualtrics.SurveyEngine.addOnPageSubmit(function() {
    if (this._aiSubmit) this._aiSubmit();
});
Qualtrics.SurveyEngine.addOnUnload(function() {
    if (this._aiCleanup) this._aiCleanup();
});
