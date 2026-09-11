from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(__file__).with_name(
    "Can_AI_select_CFD_CLIP_fillers_no_groups.qsf"
)

VERBAL_BLOCK_ID = "BL_7sK2pN4qR6tV8xY"
LINEUP_BLOCK_ID = "BL_2hONnJqBWw8pbqS"
TECHNICAL_BLOCK_ID = "BL_4cFdClIpTechFail"
VERBAL_QID = "QID59"
LINEUP_QID = "QID60"
TECHNICAL_QID = "QID61"

STUDY_VERSION = "cfd_clip_contrastive_l14_v3"
SCORING_CONTRACT_VERSION = "cfd_clip_attribute_contrastive_v3"
SCORING_CONTRACT_SHA256 = (
    "c80b630817e7f50f6ebe92b99122da3bdd9436addda153197eb5f6e8b2e897b3"
)
SELECTION_PROFILE_ID = "cfd_wm_n_pn_l14_pilot_supported_v2"
SELECTION_PROFILE_SHA256 = (
    "12e39748b8053fe3daefbe538539f171477ea421bc9c7a2f8e5f6616ddbd924f"
)
EXPECTED_MODEL_NAME = "clip-ViT-L-14"
EXPECTED_MEDIA_MANIFEST_SHA256 = (
    "dd63a355f124c49c5f505fa7ead7d776508c8081af9926d4c0ed02beb28b5d84"
)
EXPECTED_MODEL_SNAPSHOT_SHA256 = (
    "fc4ce36100d653b9871997f55bac8f100c1e43d471ec80fdad3338deb1e1bb90"
)
SIMILARITY_TYPE = "CLIP-attribute-contrastive-CFD-WM-N"


PARTICIPANT_INFORMATION_HTML = r"""<strong><h3>Is eyewitness memory better with Artificial Intelligence?</h3></strong>
<hr>

<p>I would like to invite you to take part in my research project. Before you decide whether to participate, I would like you to understand why the research is being conducted and what it would involve for you. Please <a href="mailto:damien.wang@bristol.ac.uk">email me</a> questions if anything is unclear.</p>

<p>&nbsp;</p>

<h4>What is the purpose of the project?</h4>
<p>This project examines whether an artificial-intelligence system can use an eyewitness's verbal description to select suitable lineup fillers. The system uses a pre-specified contrastive CLIP method to match supported, explicitly described appearance attributes with photographs from the Chicago Face Database (CFD).</p>

<p>&nbsp;</p>

<h4>Why have I been invited to participate?</h4>
<p>We are asking adults aged 18 or over to take part because we are looking to recruit a diverse sample of people.</p>

<p>&nbsp;</p>

<h4>Do I have to take part?</h4>
<p>It is up to you to decide whether you wish to participate in the project. Please read this information before deciding. You may ask questions before taking part. If you agree to take part, you will be asked to confirm your consent. You are free to withdraw at any time before submitting the survey, without giving a reason.</p>

<p>&nbsp;</p>

<h4>What will happen to me if I take part and what will I have to do?</h4>
<p>You will first watch a brief video (under 1 minute). You will then type a description of the person you saw. Your description will be processed by a research-hosted CLIP service to select six lineup fillers from the CFD White Male image pool. You will play Tetris for five minutes, view the lineup, make an identification decision, and rate your confidence. The experiment should take approximately 10&ndash;12 minutes.</p>

<p>&nbsp;</p>

<h4>What are the possible disadvantages and risks involved in taking part in the project?</h4>
<p>There are no known risks beyond those of ordinary computer use. Please do not include your own name, contact details, or other personal information in the description. No names will be requested, and you will be identified by a study number. The study will follow GDPR and the standards for data protection set out in the <a target="_blank" href="http://www.bristol.ac.uk/secretary/data-protection/" rel="noopener">University of Bristol guidelines</a>.</p>

<p>&nbsp;</p>

<h4>What are the possible benefits of taking part?</h4>
<p>This research is not designed to help individual participants personally, but the results may help researchers understand eyewitness identification and technology-assisted filler selection.</p>

<p>&nbsp;</p>

<h4>Will my participation in this project be kept confidential?</h4>
<p>Your participation is intended to be anonymous to the research team. Qualtrics is configured not to retain IP address or location data in the research response. The current study also loads the video from YouTube, Tetris from GitHub Pages, and lineup images from a research-hosted service. Those providers necessarily receive network metadata such as an IP address and may process it under their own privacy terms; these metadata are not included in the research dataset. The research-hosted service and its reverse proxy must be configured not to retain client IP addresses in access logs. Research data will be stored securely. De-identified data may be shared on research-data repositories; free-text descriptions must be screened for accidentally entered identifying information before any open-data release.</p>"""


CONSENT_INFORMATION_HTML = r"""<p style="font-size: 36px; color: teal;"><strong>Welcome!</strong></p>

<p>&nbsp;</p>
<p>You are invited to take part in a psychological study at the University of Bristol. Before you agree to take part, please read the following information carefully and ask me if you have any questions.</p>

<p>&nbsp;</p>
<p>You will <strong>not</strong> be able to take part on a <strong>mobile device</strong>.</p>

<p>&nbsp;</p>
<p><strong>What is the purpose of the study?</strong></p>
<p>This project examines whether CLIP-based image&ndash;text matching can use an eyewitness's verbal description to select suitable lineup fillers from the Chicago Face Database (CFD).</p>

<p>&nbsp;</p>
<p><strong>What would taking part involve?</strong></p>
<p>You will watch a brief video (under 1 minute), type a description of the person you saw, play Tetris for five minutes, view a six-person police lineup, make an identification decision, and rate your confidence. The typed description will be sent to a research-hosted CLIP service, which will return CFD image identifiers and temporary image links for the lineup.</p>

<p>&nbsp;</p>
<p>To take part, you:</p>
<p>&bull; must be <strong>18+ years old</strong>;</p>
<p>&bull; must be fluent in <strong>English</strong>;</p>
<p>&bull; must use a desktop or laptop computer.</p>

<p>&nbsp;</p>
<p>We will ask you to provide some basic demographic information. The experiment should take approximately 10&ndash;12 minutes.</p>

<p>&nbsp;</p>
<p><strong>Do I have to take part?</strong></p>
<p>It is up to you to decide whether you wish to participate. You may ask questions before taking part. If you agree, you will be asked to confirm your consent. You are free to withdraw at any time before submitting the survey, without giving a reason.</p>

<p>&nbsp;</p>
<p><strong>What happens to the data?</strong></p>
<p>Your study data will be identified by a study number rather than your name. Qualtrics is configured not to retain IP address or location data in the research response. The current study loads the video from YouTube, Tetris from GitHub Pages, and lineup images from a research-hosted service. Those providers necessarily receive network metadata such as an IP address and may process it under their own privacy terms; these metadata are not included in the research dataset. The research-hosted service and its reverse proxy must be configured not to retain client IP addresses in access logs. Research staff and authorised bodies that monitor research conduct may access the records. De-identified data may later be made open access. Before any open-data release, free-text descriptions must be checked and any accidentally entered identifying information removed.</p>

<p>&nbsp;</p>
<p><strong>Are there any benefits to taking part and what will happen to the results?</strong></p>
<p>This research is not designed to help individual participants personally, but it may improve understanding of eyewitness identification and technology-assisted filler selection.</p>

<p>&nbsp;</p>
<p><strong>Who can I contact about the study?</strong></p>
<p>If you have any questions about this study, please ask me now or <a href="mailto:damien.wang@bristol.ac.uk">email Damien Wang</a>.</p>

<p>&nbsp;</p>
<hr>
<p>&nbsp;</p>
<p><strong>General Data Protection Information (GDPR)</strong></p>
<p>Any data you provide during the study will be stored securely. Appropriate technical and organisational measures are used to prevent data from being accidentally lost, altered, disclosed, or accessed without authorisation. The principal investigator, Professor Laura Mickes, will keep the research data for 5 years after the study has finished.</p>
<p>Damien Wang and research collaborators may inspect research records to check the accuracy of the study. If the study is published, de-identified data may be made available to third parties. People analysing that information will not be able to identify you from the released dataset.</p>
<p>You can find out more about your rights under the <a href="https://www.legislation.gov.uk/ukpga/2018/12/contents/enacted">GDPR and Data Protection Act 2018</a>.</p>"""


DEBRIEF_HTML = r"""<p style="font-size: 20px; color: black;"><strong>Participant Debrief</strong>: Can a verbal description guide AI filler selection?</p>

<p>&nbsp;</p>
<p>Thank you for taking part. You watched a mock-crime video, described the person you saw, made a lineup decision, and rated your confidence. The time taken to make the lineup decision was also recorded.</p>

<p>&nbsp;</p>
<p>The lineup in this study was target-absent: all six photographs were fillers, and the correct response was therefore &ldquo;Not Present&rdquo;. We did not disclose this before the decision because knowing it would have changed the identification task.</p>

<p>&nbsp;</p>
<p>Your typed description was processed by a conservative, version-controlled parser. Only supported appearance attributes that were stated clearly and without conflicting or uncertain wording were used. For each accepted attribute, CLIP compared a positive prompt with concrete, pre-specified alternative prompts. Attribute-level contrast scores were standardised within the fixed neutral White male CFD pool and then averaged with equal weight to rank the candidates. If no supported attribute pair could be formed, the pre-specified fallback compared chunked raw description text with the same pool. The system was not given the perpetrator's identity and did not perform face recognition on the video.</p>

<p>&nbsp;</p>
<p>Police lineups contain a suspect and filler individuals who are known to be innocent. Filler choice can affect identification decisions. This single-condition study characterises correct rejections, filler identifications, confidence, and response time when fillers are selected from a verbal description. Without a comparison condition, it does not by itself establish that CLIP selection is better than another filler-selection method.</p>

<p>&nbsp;</p>
<p>If you have any questions or would like to learn more about this research, please contact <a href="mailto:damien.wang@bristol.ac.uk">Damien Wang</a>.</p>"""


TECHNICAL_FAILURE_HTML = r"""<p style="font-size: 24px; color: teal;"><strong>Thank you for taking part</strong></p>
<p>We could not load the lineup because of a technical problem. You did nothing wrong. Your response is marked as a technical failure and no identification decision from this attempt should be included in the main analysis.</p>
<p>The planned lineup was target-absent: all six photographs would have been fillers selected from the CFD White Male neutral-expression pool using the study's fixed contrastive CLIP profile. We did not disclose this earlier because it would have changed the identification task.</p>
<p>Your completion code is <strong>BS-${e://Field/RandomSurveyID}-FS</strong>.</p>
<p>If you have questions about participation or compensation, please contact <a href="mailto:damien.wang@bristol.ac.uk">Damien Wang</a> and quote the completion code above. Please do not send your verbal description by email.</p>"""


VERBAL_QUESTION_HTML = r"""<p><strong>Please describe the person you saw in the video in as much detail as you can.</strong></p>
<p>Focus on remembered appearance, such as approximate age, hair, face shape, facial features, facial hair, skin tone, and any distinctive features. If you are unsure about something, leave it out rather than guessing.</p>
<p>Write in English. Do not include your own name, contact details, or other personal information. Your description will be processed automatically to prepare the lineup.</p>
<p id="clipDescriptionValidation" aria-live="polite">Enter at least one non-space character. <span id="clipDescriptionCount">0</span>/2000 characters.</p>"""


DESCRIPTION_QUESTION_JS = r"""Qualtrics.SurveyEngine.addOnload(function() {
    this.hideNextButton();
});

Qualtrics.SurveyEngine.addOnReady(function() {
    var question = this;
    var container = this.getQuestionContainer();
    var input = container.querySelector('textarea, input[type="text"]');
    var count = document.getElementById('clipDescriptionCount');
    var validation = document.getElementById('clipDescriptionValidation');
    if (!input) return;
    input.setAttribute('maxlength', '2000');

    function update() {
        var value = String(input.value || '');
        var valid = value.trim().length > 0 && value.length <= 2000;
        if (count) count.textContent = String(value.length);
        if (validation) validation.style.color = valid ? '#2f5d46' : '#8a1c1c';
        if (valid) question.showNextButton();
        else question.hideNextButton();
    }

    input.addEventListener('input', update);
    input.addEventListener('change', update);
    update();
});

Qualtrics.SurveyEngine.addOnUnload(function() {});"""


TETRIS_PREFETCH_JS = r"""Qualtrics.SurveyEngine.addOnload(function() {
    var EXPECTED_STUDY_VERSION = 'cfd_clip_contrastive_l14_v3';
    var EXPECTED_CONTRACT_VERSION = 'cfd_clip_attribute_contrastive_v3';
    var EXPECTED_CONTRACT_HASH = 'c80b630817e7f50f6ebe92b99122da3bdd9436addda153197eb5f6e8b2e897b3';
    var EXPECTED_PROFILE_ID = 'cfd_wm_n_pn_l14_pilot_supported_v2';
    var EXPECTED_PROFILE_HASH = '12e39748b8053fe3daefbe538539f171477ea421bc9c7a2f8e5f6616ddbd924f';
    var EXPECTED_MODEL = 'clip-ViT-L-14';
    var EXPECTED_MEDIA_MANIFEST_HASH = 'dd63a355f124c49c5f505fa7ead7d776508c8081af9926d4c0ed02beb28b5d84';
    var EXPECTED_MODEL_SNAPSHOT_HASH = 'fc4ce36100d653b9871997f55bac8f100c1e43d471ec80fdad3338deb1e1bb90';
    var APPROVED_RULE_IDS = [
        'strict_v1.eyes.size.large',
        'strict_v1.eyes.size.small',
        'strict_v1.nose.length.long',
        'strict_v1.nose.length.short',
        'strict_v1.nose.width.broad',
        'strict_v1.nose.width.narrow',
        'strict_v1.lips.fullness.thin',
        'strict_v1.lips.fullness.full',
        'strict_v1.chin.length.long',
        'strict_v1.chin.length.short',
        'strict_v1.forehead.height.high',
        'strict_v1.forehead.height.low'
    ];
    var apiUrl = String(Qualtrics.SurveyEngine.getEmbeddedData('CLIP_API_URL') || '').trim();
    var description = String(Qualtrics.SurveyEngine.getEmbeddedData('verbalDescription') || '').trim();
    var participantId = String(Qualtrics.SurveyEngine.getEmbeddedData('lineupSeed') || Qualtrics.SurveyEngine.getEmbeddedData('RandomSurveyID') || 'anonymous');
    var studyVersion = String(Qualtrics.SurveyEngine.getEmbeddedData('studyVersion') || EXPECTED_STUDY_VERSION);
    var cacheKey = [
        'cfdClipLineup', participantId, studyVersion, EXPECTED_CONTRACT_VERSION,
        EXPECTED_CONTRACT_HASH, EXPECTED_PROFILE_ID, EXPECTED_PROFILE_HASH,
        EXPECTED_MEDIA_MANIFEST_HASH, EXPECTED_MODEL_SNAPSHOT_HASH
    ].join(':');
    var preloadImages = [];

    function setField(name, value) {
        Qualtrics.SurveyEngine.setEmbeddedData(name, value === undefined || value === null ? '' : String(value));
    }

    function listField(value) {
        if (value === undefined || value === null) return '';
        return Array.isArray(value) ? JSON.stringify(value) : String(value);
    }

    function isSha256(value) {
        return /^[0-9a-f]{64}$/i.test(String(value || ''));
    }

    function isNonNegativeInteger(value) {
        if (value === '' || value === null || value === undefined) return false;
        var numeric = Number(value);
        return isFinite(numeric) && numeric >= 0 && Math.floor(numeric) === numeric;
    }

    function isUniqueStringArray(value) {
        if (!Array.isArray(value)) return false;
        var seen = [];
        for (var index = 0; index < value.length; index += 1) {
            var item = value[index];
            if (typeof item !== 'string' || !item || item !== item.trim() ||
                    seen.indexOf(item) !== -1) {
                return false;
            }
            seen.push(item);
        }
        return true;
    }

    function acceptedContrastProblem(data) {
        if (!isUniqueStringArray(data.accepted_rule_ids) ||
                !isUniqueStringArray(data.accepted_dimensions)) {
            return 'The CLIP accepted-rule audit arrays are invalid.';
        }
        for (var index = 0; index < data.accepted_rule_ids.length; index += 1) {
            if (APPROVED_RULE_IDS.indexOf(data.accepted_rule_ids[index]) === -1) {
                return 'The CLIP response used a rule outside the frozen live allowlist.';
            }
        }
        if (data.accepted_rule_ids.length !== Number(data.contrast_pair_count) ||
                data.accepted_dimensions.length !== Number(data.contrast_dimension_count)) {
            return 'The CLIP accepted-rule audit counts are inconsistent.';
        }
        if (data.fallback_used) {
            if (data.accepted_rule_ids.length !== 0 || data.accepted_dimensions.length !== 0 ||
                    Number(data.contrast_pair_count) !== 0 ||
                    Number(data.contrast_dimension_count) !== 0) {
                return 'The CLIP fallback response contains accepted contrast attributes.';
            }
        } else if (data.accepted_rule_ids.length < 1 || data.accepted_dimensions.length < 1) {
            return 'The CLIP contrastive response contains no accepted attributes.';
        }
        return '';
    }

    function responseProblem(data) {
        if (!data || !Array.isArray(data.items) || data.items.length !== 6) {
            return 'The CLIP service did not return a six-person lineup.';
        }
        if (String(data.study_version || '') !== EXPECTED_STUDY_VERSION) {
            return 'The CLIP service study version does not match this survey.';
        }
        if (String(data.scoring_contract_version || '') !== EXPECTED_CONTRACT_VERSION ||
                String(data.scoring_contract_sha256 || '') !== EXPECTED_CONTRACT_HASH) {
            return 'The CLIP scoring contract does not match this survey.';
        }
        if (String(data.selection_profile_id || '') !== EXPECTED_PROFILE_ID ||
                String(data.selection_profile_sha256 || '') !== EXPECTED_PROFILE_HASH) {
            return 'The CLIP selection profile does not match this survey.';
        }
        if (!String(data.profile_status || '').trim()) {
            return 'The CLIP selection profile status is missing.';
        }
        if (String(data.model_name || '') !== EXPECTED_MODEL) {
            return 'The CLIP model does not match this survey.';
        }
        if (String(data.media_manifest_sha256 || '') !== EXPECTED_MEDIA_MANIFEST_HASH) {
            return 'The CFD media manifest does not match this survey.';
        }
        if (String(data.model_snapshot_sha256 || '') !== EXPECTED_MODEL_SNAPSHOT_HASH ||
                data.model_snapshot_match_verified !== true) {
            return 'The CLIP model snapshot does not match this survey.';
        }
        if (String(data.scoring_method || '') !== 'attribute_contrastive' ||
                String(data.aggregation_version || '') !== 'within_pool_z_equal_dimension_mean_v1' ||
                String(data.pool_profile || '') !== 'CFD-3.0|White|Male|Neutral' ||
                data.standardise_components !== true) {
            return 'The CLIP scoring method does not match this survey.';
        }
        if (data.image_text_revision_match_verified !== true) {
            return 'The CLIP image/text model revision was not verified.';
        }
        if (data.fallback_used !== true && data.fallback_used !== false) {
            return 'The CLIP fallback audit flag is invalid.';
        }
        if ((data.fallback_used && String(data.effective_method || '') !== 'raw_positive') ||
                (!data.fallback_used && String(data.effective_method || '') !== 'attribute_contrastive') ||
                (data.fallback_used && String(data.score_space || '') !==
                    'within_pool_zscore_raw_positive_cosine') ||
                (!data.fallback_used && String(data.score_space || '') !==
                    'within_pool_zscore_equal_dimension_mean') ||
                (data.fallback_used && !String(data.fallback_reason || '').trim())) {
            return 'The CLIP effective method and fallback audit are inconsistent.';
        }
        if (!/^(ready|partial|abstain)$/.test(String(data.plan_status || '')) ||
                !String(data.scorer_version || '').trim() ||
                !String(data.parser_version || '').trim() ||
                !String(data.prompt_template_version || '').trim() ||
                !String(data.query_hash_kind || '').trim() ||
                !String(data.model_revision || '').trim() ||
                !isSha256(data.index_sha256) || !isSha256(data.pool_sha256) ||
                !isSha256(data.candidate_embedding_sha256) ||
                !isSha256(data.query_sha256) || !isSha256(data.effective_query_sha256) ||
                !isSha256(data.prompt_set_sha256) || !isSha256(data.contrast_plan_sha256) ||
                !isSha256(data.schema_sha256) || !isSha256(data.prompt_template_sha256) ||
                !isNonNegativeInteger(data.prompt_count) || Number(data.prompt_count) < 1 ||
                !isNonNegativeInteger(data.contrast_pair_count) ||
                !isNonNegativeInteger(data.contrast_dimension_count) ||
                !isNonNegativeInteger(data.abstention_count)) {
            return 'The CLIP response provenance is incomplete or invalid.';
        }
        var contrastProblem = acceptedContrastProblem(data);
        if (contrastProblem) return contrastProblem;
        var seenImageIds = [];
        for (var index = 0; index < data.items.length; index += 1) {
            var item = data.items[index];
            var imageId = item ? String(item.image_id || '').trim() : '';
            var rank = item ? Number(item.selector_rank) : NaN;
            var score = item ? Number(item.selector_score) : NaN;
            if (!item || !imageId || !item.image_url || item.selector_score === '' ||
                    item.selector_score === null || item.selector_score === undefined ||
                    !isFinite(rank) || rank < 1 ||
                    Math.floor(rank) !== rank || !isFinite(score)) {
                return 'The CLIP service returned an invalid lineup member.';
            }
            if (seenImageIds.indexOf(imageId) !== -1) {
                return 'The CLIP service returned duplicate lineup members.';
            }
            seenImageIds.push(imageId);
        }
        return '';
    }

    function validResponse(data) {
        return responseProblem(data) === '';
    }

    function persist(data) {
        var problem = responseProblem(data);
        if (problem) throw new Error(problem);
        try {
            sessionStorage.setItem(cacheKey, JSON.stringify(data));
        } catch (ignore) {}

        setField('clipStatus', 'ready');
        setField('clipError', '');
        setField('clipRequestID', data.lineup_id || data.request_id || '');
        setField('clipModel', data.model_name);
        setField('clipModelRevision', data.model_revision);
        setField('clipModelSnapshotHash', data.model_snapshot_sha256);
        setField('clipModelSnapshotMatchVerified', data.model_snapshot_match_verified);
        setField('clipMediaManifestHash', data.media_manifest_sha256);
        setField('clipIndexHash', data.index_sha256);
        setField('clipCandidateEmbeddingHash', data.candidate_embedding_sha256);
        setField('clipImageTextRevisionMatchVerified', data.image_text_revision_match_verified);
        setField('clipQueryHash', data.query_sha256);
        setField('clipQueryHashKind', data.query_hash_kind);
        setField('clipPromptVersion', data.prompt_version);
        setField('clipPromptSetHash', data.prompt_set_sha256);
        setField('clipPromptCount', data.prompt_count);
        setField('clipPoolVersion', data.pool_version);
        setField('clipPoolHash', data.pool_sha256);
        setField('clipPoolProfile', data.pool_profile);
        setField('clipSelectionAlgorithm', data.selection_algorithm);
        setField('clipPoolSize', data.pool_size);
        setField('clipServiceStudyVersion', data.study_version);
        setField('clipQueryStrategy', data.query_strategy);
        setField('clipQueryChunkCount', data.query_chunk_count);
        setField('clipQueryTokenCount', data.query_token_count);
        setField('clipContextTokens', data.model_context_tokens);
        setField('clipEffectiveQueryHash', data.effective_query_sha256);
        setField('clipScoringContractVersion', data.scoring_contract_version);
        setField('clipScoringContractHash', data.scoring_contract_sha256);
        setField('clipSelectionProfileID', data.selection_profile_id);
        setField('clipSelectionProfileVersion', data.selection_profile_version);
        setField('clipSelectionProfileHash', data.selection_profile_sha256);
        setField('clipSelectionProfileStatus', data.profile_status);
        setField('clipScoringMethod', data.scoring_method);
        setField('clipEffectiveMethod', data.effective_method);
        setField('clipScoreSpace', data.score_space);
        setField('clipAggregationVersion', data.aggregation_version);
        setField('clipScorerVersion', data.scorer_version);
        setField('clipPlanStatus', data.plan_status);
        setField('clipContrastPlanHash', data.contrast_plan_sha256);
        setField('clipContrastSchemaName', data.schema_name);
        setField('clipContrastSchemaVersion', data.schema_version);
        setField('clipContrastSchemaHash', data.schema_sha256);
        setField('clipParserVersion', data.parser_version);
        setField('clipPromptTemplateVersion', data.prompt_template_version);
        setField('clipPromptTemplateHash', data.prompt_template_sha256);
        setField('clipContrastPairCount', data.contrast_pair_count);
        setField('clipContrastDimensionCount', data.contrast_dimension_count);
        setField('clipAcceptedRuleIDs', listField(data.accepted_rule_ids));
        setField('clipAcceptedDimensions', listField(data.accepted_dimensions));
        setField('clipAbstentionCount', data.abstention_count);
        setField('clipAbstentionReasonCodes', listField(data.abstention_reason_codes));
        setField('clipFallbackUsed', data.fallback_used);
        setField('clipFallbackReason', data.fallback_reason);
        setField('clipStandardiseComponents', data.standardise_components);

        data.items.forEach(function(item, index) {
            var position = index + 1;
            var absoluteUrl = new URL(item.image_url, apiUrl).href;
            setField('lineup' + position, item.image_id);
            setField('lineup' + position + 'Url', absoluteUrl);
            setField('lineup' + position + 'Rank', item.selector_rank);
            setField('lineup' + position + 'Score', item.selector_score);
            var preloader = new Image();
            preloader.referrerPolicy = 'no-referrer';
            preloader.src = absoluteUrl;
            preloadImages.push(preloader);
        });
        setField('lineupPreloadStarted', 'true');
    }

    function fail(error) {
        var attempts = parseInt(Qualtrics.SurveyEngine.getEmbeddedData('technicalFailureAttempts') || '0', 10) || 0;
        setField('technicalFailureAttempts', attempts + 1);
        setField('clipStatus', 'error');
        setField('clipError', String(error && error.message ? error.message : error).slice(0, 250));
    }

    if (studyVersion !== EXPECTED_STUDY_VERSION) {
        fail(new Error('The configured study version does not match this survey build.'));
        return;
    }
    try {
        var cached = sessionStorage.getItem(cacheKey);
        if (cached) {
            var parsed = JSON.parse(cached);
            if (validResponse(parsed)) {
                persist(parsed);
                return;
            }
            sessionStorage.removeItem(cacheKey);
        }
    } catch (ignore) {}

    if (!description) {
        fail(new Error('The verbal description is missing.'));
        return;
    }
    if (!/^https:\/\//i.test(apiUrl) || apiUrl.indexOf('YOUR-CLIP-SERVICE') !== -1) {
        fail(new Error('CLIP_API_URL has not been configured.'));
        return;
    }

    setField('clipStatus', 'loading');
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var requestOptions = {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            participant_id: participantId,
            description: description,
            study_version: studyVersion
        })
    };
    if (controller) requestOptions.signal = controller.signal;
    var timeoutId;
    var timeoutPromise = new Promise(function(resolve, reject) {
        timeoutId = setTimeout(function() {
            if (controller) controller.abort();
            reject(new Error('CLIP service request timed out after 30 seconds.'));
        }, 30000);
    });
    var requestPromise = fetch(apiUrl, requestOptions).then(function(response) {
        if (!response.ok) {
            throw new Error('CLIP service returned HTTP ' + response.status + '.');
        }
        return response.json();
    });
    Promise.race([requestPromise, timeoutPromise]).then(function(data) {
        clearTimeout(timeoutId);
        return data;
    }, function(error) {
        clearTimeout(timeoutId);
        if (error && error.name === 'AbortError') {
            throw new Error('CLIP service request timed out after 30 seconds.');
        }
        throw error;
    }).then(persist).catch(fail);
});

Qualtrics.SurveyEngine.addOnReady(function() {});
Qualtrics.SurveyEngine.addOnUnload(function() {});"""


LINEUP_QUESTION_JS = r"""Qualtrics.SurveyEngine.addOnload(function() {
    this.hideNextButton();
    Qualtrics.SurveyEngine.setEmbeddedData('SelectionMade', 'false');
});

Qualtrics.SurveyEngine.addOnReady(function() {
    var question = this;
    var container = this.getQuestionContainer();
    var EXPECTED_STUDY_VERSION = 'cfd_clip_contrastive_l14_v3';
    var EXPECTED_CONTRACT_VERSION = 'cfd_clip_attribute_contrastive_v3';
    var EXPECTED_CONTRACT_HASH = 'c80b630817e7f50f6ebe92b99122da3bdd9436addda153197eb5f6e8b2e897b3';
    var EXPECTED_PROFILE_ID = 'cfd_wm_n_pn_l14_pilot_supported_v2';
    var EXPECTED_PROFILE_HASH = '12e39748b8053fe3daefbe538539f171477ea421bc9c7a2f8e5f6616ddbd924f';
    var EXPECTED_MODEL = 'clip-ViT-L-14';
    var EXPECTED_MEDIA_MANIFEST_HASH = 'dd63a355f124c49c5f505fa7ead7d776508c8081af9926d4c0ed02beb28b5d84';
    var EXPECTED_MODEL_SNAPSHOT_HASH = 'fc4ce36100d653b9871997f55bac8f100c1e43d471ec80fdad3338deb1e1bb90';
    var APPROVED_RULE_IDS = [
        'strict_v1.eyes.size.large',
        'strict_v1.eyes.size.small',
        'strict_v1.nose.length.long',
        'strict_v1.nose.length.short',
        'strict_v1.nose.width.broad',
        'strict_v1.nose.width.narrow',
        'strict_v1.lips.fullness.thin',
        'strict_v1.lips.fullness.full',
        'strict_v1.chin.length.long',
        'strict_v1.chin.length.short',
        'strict_v1.forehead.height.high',
        'strict_v1.forehead.height.low'
    ];
    var apiUrl = String(Qualtrics.SurveyEngine.getEmbeddedData('CLIP_API_URL') || '').trim();
    var description = String(Qualtrics.SurveyEngine.getEmbeddedData('verbalDescription') || '').trim();
    var participantId = String(Qualtrics.SurveyEngine.getEmbeddedData('lineupSeed') || Qualtrics.SurveyEngine.getEmbeddedData('RandomSurveyID') || 'anonymous');
    var studyVersion = String(Qualtrics.SurveyEngine.getEmbeddedData('studyVersion') || EXPECTED_STUDY_VERSION);
    var cacheKey = [
        'cfdClipLineup', participantId, studyVersion, EXPECTED_CONTRACT_VERSION,
        EXPECTED_CONTRACT_HASH, EXPECTED_PROFILE_ID, EXPECTED_PROFILE_HASH,
        EXPECTED_MEDIA_MANIFEST_HASH, EXPECTED_MODEL_SNAPSHOT_HASH
    ].join(':');
    var confidenceQuestion = document.getElementById('QID38');
    var statusNode = document.getElementById('clipLineupStatus');
    var retryButton = document.getElementById('clipLineupRetry');
    var responseInputs = Array.prototype.slice.call(container.querySelectorAll('input[type="radio"]'));
    var currentItems = [];
    var pageReadyAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    var lineupReadyAt = null;
    var firstDecisionRecorded = false;
    var failureAttempts = parseInt(Qualtrics.SurveyEngine.getEmbeddedData('technicalFailureAttempts') || '0', 10) || 0;
    var technicalExitInput = null;
    var technicalExitSelection = null;

    function nowMs() {
        return typeof performance !== 'undefined' ? performance.now() : Date.now();
    }

    var style = document.createElement('style');
    style.textContent = [
        '#QID60 .ChoiceStructure{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;max-width:780px;margin:0 auto!important}',
        '#QID60 .ChoiceStructure>tbody,#QID60 .ChoiceStructure>tbody>tr{display:contents!important}',
        '#QID60 .Selection{position:relative;box-sizing:border-box;border:2px solid #d3d7dc;border-radius:6px;padding:8px!important;margin:0!important;background:#fff;min-width:0}',
        '#QID60 .Selection.clip-wide-choice{grid-column:1/-1;text-align:center;padding:12px!important}',
        '#QID60 .Selection.clip-selected{border-color:#16784a;box-shadow:0 0 0 2px rgba(22,120,74,.18)}',
        '#QID60 .Selection input[type=radio]{position:absolute;opacity:0}',
        '#QID60 .Selection label{display:block;width:100%;cursor:pointer;text-align:center}',
        '#QID60 .Selection input[type=radio]:focus+label{outline:3px solid #2b6cb0;outline-offset:3px}',
        '#QID60 .clip-choice-content{display:flex;flex-direction:column;gap:6px;align-items:center}',
        '#QID60 .clip-lineup-image{display:block;width:100%;aspect-ratio:1/1.25;object-fit:cover;background:#eef1f4}',
        '#QID60 .clip-not-present{display:block;font-weight:700}',
        '#clipLineupStatus{max-width:780px;margin:0 auto 14px;padding:10px;border-radius:4px;background:#eef4f8}',
        '#clipLineupStatus[data-state=error]{background:#fff0f0;color:#8a1c1c}',
        '#clipLineupRetry{margin-left:8px}',
        '@media(max-width:650px){#QID60 .ChoiceStructure{grid-template-columns:repeat(2,minmax(0,1fr))}#QID60 .Selection.clip-wide-choice{grid-column:1/-1}}'
    ].join('');
    document.head.appendChild(style);

    function setField(name, value) {
        Qualtrics.SurveyEngine.setEmbeddedData(name, value === undefined || value === null ? '' : String(value));
    }

    function listField(value) {
        if (value === undefined || value === null) return '';
        return Array.isArray(value) ? JSON.stringify(value) : String(value);
    }

    function isSha256(value) {
        return /^[0-9a-f]{64}$/i.test(String(value || ''));
    }

    function isNonNegativeInteger(value) {
        if (value === '' || value === null || value === undefined) return false;
        var numeric = Number(value);
        return isFinite(numeric) && numeric >= 0 && Math.floor(numeric) === numeric;
    }

    function isUniqueStringArray(value) {
        if (!Array.isArray(value)) return false;
        var seen = [];
        for (var index = 0; index < value.length; index += 1) {
            var item = value[index];
            if (typeof item !== 'string' || !item || item !== item.trim() ||
                    seen.indexOf(item) !== -1) {
                return false;
            }
            seen.push(item);
        }
        return true;
    }

    function acceptedContrastProblem(data) {
        if (!isUniqueStringArray(data.accepted_rule_ids) ||
                !isUniqueStringArray(data.accepted_dimensions)) {
            return 'The CLIP accepted-rule audit arrays are invalid.';
        }
        for (var index = 0; index < data.accepted_rule_ids.length; index += 1) {
            if (APPROVED_RULE_IDS.indexOf(data.accepted_rule_ids[index]) === -1) {
                return 'The CLIP response used a rule outside the frozen live allowlist.';
            }
        }
        if (data.accepted_rule_ids.length !== Number(data.contrast_pair_count) ||
                data.accepted_dimensions.length !== Number(data.contrast_dimension_count)) {
            return 'The CLIP accepted-rule audit counts are inconsistent.';
        }
        if (data.fallback_used) {
            if (data.accepted_rule_ids.length !== 0 || data.accepted_dimensions.length !== 0 ||
                    Number(data.contrast_pair_count) !== 0 ||
                    Number(data.contrast_dimension_count) !== 0) {
                return 'The CLIP fallback response contains accepted contrast attributes.';
            }
        } else if (data.accepted_rule_ids.length < 1 || data.accepted_dimensions.length < 1) {
            return 'The CLIP contrastive response contains no accepted attributes.';
        }
        return '';
    }

    function embeddedBoolean(name) {
        return String(Qualtrics.SurveyEngine.getEmbeddedData(name) || '').toLowerCase() === 'true';
    }

    function embeddedJsonArray(name) {
        var raw = String(Qualtrics.SurveyEngine.getEmbeddedData(name) || '');
        if (!raw) return null;
        try {
            var parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : null;
        } catch (ignore) {
            return null;
        }
    }

    function setStatus(message, state, showRetry) {
        if (!statusNode) return;
        statusNode.firstChild.nodeValue = message + ' ';
        statusNode.setAttribute('data-state', state || 'loading');
        if (retryButton) retryButton.style.display = showRetry ? 'inline-block' : 'none';
    }

    function setResponsesEnabled(enabled) {
        responseInputs.forEach(function(input) { input.disabled = !enabled; });
    }

    function inputChoiceId(input) {
        var raw = input.getAttribute('choiceid') || input.getAttribute('data-choiceid') || input.value || '';
        var parts = String(raw).split('~');
        return parts[parts.length - 1];
    }

    var notPresentInput = responseInputs.filter(function(input) {
        return String(inputChoiceId(input)) === '7';
    })[0] || null;
    var notPresentSelection = notPresentInput ?
        (notPresentInput.closest ? notPresentInput.closest('.Selection') : notPresentInput.parentNode) : null;
    if (notPresentSelection) notPresentSelection.classList.add('clip-wide-choice');

    technicalExitInput = responseInputs.filter(function(input) {
        return String(inputChoiceId(input)) === '8';
    })[0] || null;
    technicalExitSelection = technicalExitInput ?
        (technicalExitInput.closest ? technicalExitInput.closest('.Selection') : technicalExitInput.parentNode) : null;
    if (technicalExitSelection) {
        technicalExitSelection.classList.add('clip-wide-choice');
        technicalExitSelection.style.display = 'none';
    }

    function hideTechnicalExit() {
        if (technicalExitInput) technicalExitInput.disabled = true;
        if (technicalExitSelection) technicalExitSelection.style.display = 'none';
    }

    function revealTechnicalExit() {
        if (technicalExitSelection) technicalExitSelection.style.display = '';
        if (technicalExitInput) technicalExitInput.disabled = false;
    }

    function clearTechnicalChoice() {
        if (typeof question.setChoiceValue === 'function') {
            question.setChoiceValue(8, false);
        }
        if (technicalExitInput) technicalExitInput.checked = false;
        var confidenceRegistry = Qualtrics.SurveyEngine.registry || {};
        var confidenceEngine = confidenceRegistry.QID38;
        if (confidenceEngine && typeof confidenceEngine.setChoiceValue === 'function') {
            confidenceEngine.setChoiceValue(1, false);
        } else {
            var confidenceInput = document.querySelector('#QID38 input[type="radio"][choiceid="1"]');
            if (confidenceInput) confidenceInput.checked = false;
        }
        setField('technicalFailure', 'false');
        setField('SelectionMade', 'false');
        setField('SelectedPosition', 'None');
        setField('SelectedImage', 'None');
        setField('selectedRetrievalRank', '');
        setField('selectedRetrievalScore', '');
        setField('selectedClipRank', '');
        setField('selectedClipScore', '');
        setField('Outcome', 'None');
        setField('responseType', 'None');
        markVisualSelection('');
        if (confidenceQuestion) confidenceQuestion.style.display = 'none';
        question.hideNextButton();
    }

    function responseProblem(data) {
        if (!data || !Array.isArray(data.items) || data.items.length !== 6) {
            return 'The CLIP service did not return a six-person lineup.';
        }
        if (String(data.study_version || '') !== EXPECTED_STUDY_VERSION) {
            return 'The CLIP service study version does not match this survey.';
        }
        if (String(data.scoring_contract_version || '') !== EXPECTED_CONTRACT_VERSION ||
                String(data.scoring_contract_sha256 || '') !== EXPECTED_CONTRACT_HASH) {
            return 'The CLIP scoring contract does not match this survey.';
        }
        if (String(data.selection_profile_id || '') !== EXPECTED_PROFILE_ID ||
                String(data.selection_profile_sha256 || '') !== EXPECTED_PROFILE_HASH) {
            return 'The CLIP selection profile does not match this survey.';
        }
        if (!String(data.profile_status || '').trim()) {
            return 'The CLIP selection profile status is missing.';
        }
        if (String(data.model_name || '') !== EXPECTED_MODEL) {
            return 'The CLIP model does not match this survey.';
        }
        if (String(data.media_manifest_sha256 || '') !== EXPECTED_MEDIA_MANIFEST_HASH) {
            return 'The CFD media manifest does not match this survey.';
        }
        if (String(data.model_snapshot_sha256 || '') !== EXPECTED_MODEL_SNAPSHOT_HASH ||
                data.model_snapshot_match_verified !== true) {
            return 'The CLIP model snapshot does not match this survey.';
        }
        if (String(data.scoring_method || '') !== 'attribute_contrastive' ||
                String(data.aggregation_version || '') !== 'within_pool_z_equal_dimension_mean_v1' ||
                String(data.pool_profile || '') !== 'CFD-3.0|White|Male|Neutral' ||
                data.standardise_components !== true) {
            return 'The CLIP scoring method does not match this survey.';
        }
        if (data.image_text_revision_match_verified !== true) {
            return 'The CLIP image/text model revision was not verified.';
        }
        if (data.fallback_used !== true && data.fallback_used !== false) {
            return 'The CLIP fallback audit flag is invalid.';
        }
        if ((data.fallback_used && String(data.effective_method || '') !== 'raw_positive') ||
                (!data.fallback_used && String(data.effective_method || '') !== 'attribute_contrastive') ||
                (data.fallback_used && String(data.score_space || '') !==
                    'within_pool_zscore_raw_positive_cosine') ||
                (!data.fallback_used && String(data.score_space || '') !==
                    'within_pool_zscore_equal_dimension_mean') ||
                (data.fallback_used && !String(data.fallback_reason || '').trim())) {
            return 'The CLIP effective method and fallback audit are inconsistent.';
        }
        if (!/^(ready|partial|abstain)$/.test(String(data.plan_status || '')) ||
                !String(data.scorer_version || '').trim() ||
                !String(data.parser_version || '').trim() ||
                !String(data.prompt_template_version || '').trim() ||
                !String(data.query_hash_kind || '').trim() ||
                !String(data.model_revision || '').trim() ||
                !isSha256(data.index_sha256) || !isSha256(data.pool_sha256) ||
                !isSha256(data.candidate_embedding_sha256) ||
                !isSha256(data.query_sha256) || !isSha256(data.effective_query_sha256) ||
                !isSha256(data.prompt_set_sha256) || !isSha256(data.contrast_plan_sha256) ||
                !isSha256(data.schema_sha256) || !isSha256(data.prompt_template_sha256) ||
                !isNonNegativeInteger(data.prompt_count) || Number(data.prompt_count) < 1 ||
                !isNonNegativeInteger(data.contrast_pair_count) ||
                !isNonNegativeInteger(data.contrast_dimension_count) ||
                !isNonNegativeInteger(data.abstention_count)) {
            return 'The CLIP response provenance is incomplete or invalid.';
        }
        var contrastProblem = acceptedContrastProblem(data);
        if (contrastProblem) return contrastProblem;
        var seenImageIds = [];
        for (var index = 0; index < data.items.length; index += 1) {
            var item = data.items[index];
            var imageId = item ? String(item.image_id || '').trim() : '';
            var rank = item ? Number(item.selector_rank) : NaN;
            var score = item ? Number(item.selector_score) : NaN;
            if (!item || !imageId || !item.image_url || item.selector_score === '' ||
                    item.selector_score === null || item.selector_score === undefined ||
                    !isFinite(rank) || rank < 1 ||
                    Math.floor(rank) !== rank || !isFinite(score)) {
                return 'The CLIP service returned an invalid lineup member.';
            }
            if (seenImageIds.indexOf(imageId) !== -1) {
                return 'The CLIP service returned duplicate lineup members.';
            }
            seenImageIds.push(imageId);
        }
        return '';
    }

    function normaliseData(data) {
        var problem = responseProblem(data);
        if (problem) throw new Error(problem);
        data.items = data.items.map(function(item, index) {
            var copy = Object.assign({}, item);
            copy.position = index + 1;
            copy.image_url = new URL(copy.image_url, apiUrl).href;
            return copy;
        });
        return data;
    }

    function persist(data) {
        data = normaliseData(data);
        currentItems = data.items;
        try { sessionStorage.setItem(cacheKey, JSON.stringify(data)); } catch (ignore) {}
        setField('clipStatus', 'ready');
        setField('clipError', '');
        setField('clipRequestID', data.lineup_id || data.request_id || '');
        setField('clipModel', data.model_name);
        setField('clipModelRevision', data.model_revision);
        setField('clipModelSnapshotHash', data.model_snapshot_sha256);
        setField('clipModelSnapshotMatchVerified', data.model_snapshot_match_verified);
        setField('clipMediaManifestHash', data.media_manifest_sha256);
        setField('clipIndexHash', data.index_sha256);
        setField('clipCandidateEmbeddingHash', data.candidate_embedding_sha256);
        setField('clipImageTextRevisionMatchVerified', data.image_text_revision_match_verified);
        setField('clipQueryHash', data.query_sha256);
        setField('clipQueryHashKind', data.query_hash_kind);
        setField('clipPromptVersion', data.prompt_version);
        setField('clipPromptSetHash', data.prompt_set_sha256);
        setField('clipPromptCount', data.prompt_count);
        setField('clipPoolVersion', data.pool_version);
        setField('clipPoolHash', data.pool_sha256);
        setField('clipPoolProfile', data.pool_profile);
        setField('clipSelectionAlgorithm', data.selection_algorithm);
        setField('clipPoolSize', data.pool_size);
        setField('clipServiceStudyVersion', data.study_version);
        setField('clipQueryStrategy', data.query_strategy);
        setField('clipQueryChunkCount', data.query_chunk_count);
        setField('clipQueryTokenCount', data.query_token_count);
        setField('clipContextTokens', data.model_context_tokens);
        setField('clipEffectiveQueryHash', data.effective_query_sha256);
        setField('clipScoringContractVersion', data.scoring_contract_version);
        setField('clipScoringContractHash', data.scoring_contract_sha256);
        setField('clipSelectionProfileID', data.selection_profile_id);
        setField('clipSelectionProfileVersion', data.selection_profile_version);
        setField('clipSelectionProfileHash', data.selection_profile_sha256);
        setField('clipSelectionProfileStatus', data.profile_status);
        setField('clipScoringMethod', data.scoring_method);
        setField('clipEffectiveMethod', data.effective_method);
        setField('clipScoreSpace', data.score_space);
        setField('clipAggregationVersion', data.aggregation_version);
        setField('clipScorerVersion', data.scorer_version);
        setField('clipPlanStatus', data.plan_status);
        setField('clipContrastPlanHash', data.contrast_plan_sha256);
        setField('clipContrastSchemaName', data.schema_name);
        setField('clipContrastSchemaVersion', data.schema_version);
        setField('clipContrastSchemaHash', data.schema_sha256);
        setField('clipParserVersion', data.parser_version);
        setField('clipPromptTemplateVersion', data.prompt_template_version);
        setField('clipPromptTemplateHash', data.prompt_template_sha256);
        setField('clipContrastPairCount', data.contrast_pair_count);
        setField('clipContrastDimensionCount', data.contrast_dimension_count);
        setField('clipAcceptedRuleIDs', listField(data.accepted_rule_ids));
        setField('clipAcceptedDimensions', listField(data.accepted_dimensions));
        setField('clipAbstentionCount', data.abstention_count);
        setField('clipAbstentionReasonCodes', listField(data.abstention_reason_codes));
        setField('clipFallbackUsed', data.fallback_used);
        setField('clipFallbackReason', data.fallback_reason);
        setField('clipStandardiseComponents', data.standardise_components);
        currentItems.forEach(function(item, index) {
            var position = index + 1;
            setField('lineup' + position, item.image_id);
            setField('lineup' + position + 'Url', item.image_url);
            setField('lineup' + position + 'Rank', item.selector_rank);
            setField('lineup' + position + 'Score', item.selector_score);
        });
        return data;
    }

    function embeddedData() {
        var items = [];
        for (var position = 1; position <= 6; position += 1) {
            var imageId = String(Qualtrics.SurveyEngine.getEmbeddedData('lineup' + position) || '').trim();
            var imageUrl = String(Qualtrics.SurveyEngine.getEmbeddedData('lineup' + position + 'Url') || '').trim();
            if (!imageId || !imageUrl) return null;
            items.push({
                position: position,
                image_id: imageId,
                image_url: imageUrl,
                selector_rank: Qualtrics.SurveyEngine.getEmbeddedData('lineup' + position + 'Rank'),
                selector_score: Qualtrics.SurveyEngine.getEmbeddedData('lineup' + position + 'Score')
            });
        }
        if (String(Qualtrics.SurveyEngine.getEmbeddedData('clipStatus') || '') !== 'ready') {
            return null;
        }
        return {
            items: items,
            study_version: Qualtrics.SurveyEngine.getEmbeddedData('clipServiceStudyVersion'),
            scoring_contract_version: Qualtrics.SurveyEngine.getEmbeddedData('clipScoringContractVersion'),
            scoring_contract_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipScoringContractHash'),
            selection_profile_id: Qualtrics.SurveyEngine.getEmbeddedData('clipSelectionProfileID'),
            selection_profile_version: Qualtrics.SurveyEngine.getEmbeddedData('clipSelectionProfileVersion'),
            selection_profile_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipSelectionProfileHash'),
            profile_status: Qualtrics.SurveyEngine.getEmbeddedData('clipSelectionProfileStatus'),
            model_name: Qualtrics.SurveyEngine.getEmbeddedData('clipModel'),
            model_revision: Qualtrics.SurveyEngine.getEmbeddedData('clipModelRevision'),
            model_snapshot_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipModelSnapshotHash'),
            model_snapshot_match_verified: embeddedBoolean('clipModelSnapshotMatchVerified'),
            media_manifest_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipMediaManifestHash'),
            index_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipIndexHash'),
            candidate_embedding_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipCandidateEmbeddingHash'),
            image_text_revision_match_verified: embeddedBoolean('clipImageTextRevisionMatchVerified'),
            query_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipQueryHash'),
            query_hash_kind: Qualtrics.SurveyEngine.getEmbeddedData('clipQueryHashKind'),
            prompt_version: Qualtrics.SurveyEngine.getEmbeddedData('clipPromptVersion'),
            prompt_set_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipPromptSetHash'),
            prompt_count: Qualtrics.SurveyEngine.getEmbeddedData('clipPromptCount'),
            pool_version: Qualtrics.SurveyEngine.getEmbeddedData('clipPoolVersion'),
            pool_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipPoolHash'),
            pool_profile: Qualtrics.SurveyEngine.getEmbeddedData('clipPoolProfile'),
            selection_algorithm: Qualtrics.SurveyEngine.getEmbeddedData('clipSelectionAlgorithm'),
            pool_size: Qualtrics.SurveyEngine.getEmbeddedData('clipPoolSize'),
            query_strategy: Qualtrics.SurveyEngine.getEmbeddedData('clipQueryStrategy'),
            query_chunk_count: Qualtrics.SurveyEngine.getEmbeddedData('clipQueryChunkCount'),
            query_token_count: Qualtrics.SurveyEngine.getEmbeddedData('clipQueryTokenCount'),
            model_context_tokens: Qualtrics.SurveyEngine.getEmbeddedData('clipContextTokens'),
            effective_query_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipEffectiveQueryHash'),
            scoring_method: Qualtrics.SurveyEngine.getEmbeddedData('clipScoringMethod'),
            effective_method: Qualtrics.SurveyEngine.getEmbeddedData('clipEffectiveMethod'),
            score_space: Qualtrics.SurveyEngine.getEmbeddedData('clipScoreSpace'),
            aggregation_version: Qualtrics.SurveyEngine.getEmbeddedData('clipAggregationVersion'),
            scorer_version: Qualtrics.SurveyEngine.getEmbeddedData('clipScorerVersion'),
            plan_status: Qualtrics.SurveyEngine.getEmbeddedData('clipPlanStatus'),
            contrast_plan_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastPlanHash'),
            schema_name: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastSchemaName'),
            schema_version: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastSchemaVersion'),
            schema_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastSchemaHash'),
            parser_version: Qualtrics.SurveyEngine.getEmbeddedData('clipParserVersion'),
            prompt_template_version: Qualtrics.SurveyEngine.getEmbeddedData('clipPromptTemplateVersion'),
            prompt_template_sha256: Qualtrics.SurveyEngine.getEmbeddedData('clipPromptTemplateHash'),
            contrast_pair_count: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastPairCount'),
            contrast_dimension_count: Qualtrics.SurveyEngine.getEmbeddedData('clipContrastDimensionCount'),
            accepted_rule_ids: embeddedJsonArray('clipAcceptedRuleIDs'),
            accepted_dimensions: embeddedJsonArray('clipAcceptedDimensions'),
            abstention_count: Qualtrics.SurveyEngine.getEmbeddedData('clipAbstentionCount'),
            abstention_reason_codes: Qualtrics.SurveyEngine.getEmbeddedData('clipAbstentionReasonCodes'),
            fallback_used: embeddedBoolean('clipFallbackUsed'),
            fallback_reason: Qualtrics.SurveyEngine.getEmbeddedData('clipFallbackReason'),
            standardise_components: embeddedBoolean('clipStandardiseComponents')
        };
    }

    function showImages(data) {
        data = normaliseData(data);
        currentItems = data.items;
        var loads = currentItems.map(function(item, index) {
            return new Promise(function(resolve, reject) {
                var image = container.querySelector('.clip-lineup-image[data-position="' + (index + 1) + '"]');
                if (!image) {
                    reject(new Error('Lineup image element ' + (index + 1) + ' is missing.'));
                    return;
                }
                var imageTimeout = setTimeout(function() {
                    reject(new Error('Lineup image ' + (index + 1) + ' timed out after 20 seconds.'));
                }, 20000);
                image.onload = function() {
                    clearTimeout(imageTimeout);
                    resolve();
                };
                image.onerror = function() {
                    clearTimeout(imageTimeout);
                    reject(new Error('Lineup image ' + (index + 1) + ' could not be loaded.'));
                };
                image.referrerPolicy = 'no-referrer';
                image.src = item.image_url;
            });
        });
        return Promise.all(loads).then(function() {
            clearTechnicalChoice();
            lineupReadyAt = nowMs();
            setField('lineupLoadTimeMs', Math.round(lineupReadyAt - pageReadyAt));
            setStatus('Select one face, or choose Not Present.', 'ready', false);
            setResponsesEnabled(true);
            hideTechnicalExit();
        });
    }

    function fetchLineup() {
        clearTechnicalChoice();
        hideTechnicalExit();
        setResponsesEnabled(false);
        setStatus('Preparing your lineup...', 'loading', false);
        if (!description) return Promise.reject(new Error('The verbal description is missing.'));
        if (!/^https:\/\//i.test(apiUrl) || apiUrl.indexOf('YOUR-CLIP-SERVICE') !== -1) {
            return Promise.reject(new Error('CLIP_API_URL has not been configured in Survey Flow.'));
        }
        setField('clipStatus', 'loading');
        var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
        var requestOptions = {
            method: 'POST',
            mode: 'cors',
            credentials: 'omit',
            referrerPolicy: 'no-referrer',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                participant_id: participantId,
                description: description,
                study_version: studyVersion
            })
        };
        if (controller) requestOptions.signal = controller.signal;
        var timeoutId;
        var timeoutPromise = new Promise(function(resolve, reject) {
            timeoutId = setTimeout(function() {
                if (controller) controller.abort();
                reject(new Error('CLIP service request timed out after 30 seconds.'));
            }, 30000);
        });
        var requestPromise = fetch(apiUrl, requestOptions).then(function(response) {
            if (!response.ok) throw new Error('CLIP service returned HTTP ' + response.status + '.');
            return response.json();
        });
        return Promise.race([requestPromise, timeoutPromise]).then(function(data) {
            clearTimeout(timeoutId);
            return data;
        }, function(error) {
            clearTimeout(timeoutId);
            if (error && error.name === 'AbortError') {
                throw new Error('CLIP service request timed out after 30 seconds.');
            }
            throw error;
        }).then(persist).then(showImages);
    }

    function fail(error) {
        var message = String(error && error.message ? error.message : error).slice(0, 250);
        failureAttempts += 1;
        setField('technicalFailureAttempts', failureAttempts);
        setField('technicalFailureReason', message);
        setField('clipStatus', 'error');
        setField('clipError', message);
        setResponsesEnabled(false);
        if (failureAttempts >= 3) revealTechnicalExit();
        setStatus(
            'The lineup could not be prepared. Please use Retry.' +
            (failureAttempts >= 3 ? ' You may instead end the study using the technical-problem option below.' : '') +
            ' (' + message + ')',
            'error',
            true
        );
    }

    function markVisualSelection(choice) {
        Array.prototype.slice.call(container.querySelectorAll('.Selection')).forEach(function(node) {
            node.classList.remove('clip-selected');
        });
        var input = responseInputs.filter(function(candidate) {
            return String(inputChoiceId(candidate)) === String(choice);
        })[0];
        if (input) {
            var selection = input.closest ? input.closest('.Selection') : input.parentNode;
            if (selection) selection.classList.add('clip-selected');
        }
    }

    function recordChoice(choice) {
        var numericChoice = parseInt(choice, 10);
        if (numericChoice >= 1 && numericChoice <= 6) {
            var item = currentItems[numericChoice - 1];
            if (!item) return;
            setField('SelectedPosition', numericChoice);
            setField('SelectedImage', item.image_id);
            setField('selectedRetrievalRank', item.selector_rank);
            setField('selectedRetrievalScore', item.selector_score);
            setField('selectedClipRank', item.selector_rank);
            setField('selectedClipScore', item.selector_score);
            setField('Outcome', 'Filler ID');
            setField('responseType', 'fillerId');
        } else if (numericChoice === 7) {
            setField('SelectedPosition', 'NotPresent');
            setField('SelectedImage', 'None');
            setField('selectedRetrievalRank', '');
            setField('selectedRetrievalScore', '');
            setField('selectedClipRank', '');
            setField('selectedClipScore', '');
            setField('Outcome', 'Correct Rejection');
            setField('responseType', 'rejectId');
        } else if (numericChoice === 8 && failureAttempts >= 3) {
            setField('SelectedPosition', 'TechnicalFailure');
            setField('SelectedImage', 'None');
            setField('selectedRetrievalRank', '');
            setField('selectedRetrievalScore', '');
            setField('selectedClipRank', '');
            setField('selectedClipScore', '');
            setField('Outcome', 'Technical Failure');
            setField('responseType', 'technicalFailure');
            setField('technicalFailure', 'true');
            var confidenceRegistry = Qualtrics.SurveyEngine.registry || {};
            var confidenceEngine = confidenceRegistry.QID38;
            if (confidenceEngine && typeof confidenceEngine.setChoiceValue === 'function') {
                confidenceEngine.setChoiceValue(1, true);
            } else {
                var confidenceInput = document.querySelector('#QID38 input[type="radio"][choiceid="1"]');
                if (confidenceInput) confidenceInput.click();
            }
        } else {
            return;
        }
        if (numericChoice <= 7 && !firstDecisionRecorded && lineupReadyAt !== null) {
            var decisionMs = Math.max(0, Math.round(nowMs() - lineupReadyAt));
            setField('LineupDecisionRTms', decisionMs);
            setField('LineupDecisionRTs', (decisionMs / 1000).toFixed(3));
            firstDecisionRecorded = true;
        }
        setField('SelectionMade', 'true');
        markVisualSelection(numericChoice);
        if (numericChoice <= 7 && confidenceQuestion) confidenceQuestion.style.display = 'block';
        question.showNextButton();
    }

    responseInputs.forEach(function(input) {
        input.addEventListener('change', function() {
            if (input.checked) recordChoice(inputChoiceId(input));
        });
    });

    question.questionclick = function() {
        var selected = question.getSelectedChoices();
        if (selected && selected.length) recordChoice(selected[0]);
    };
    question._clipRecordChoice = recordChoice;

    if (retryButton) {
        retryButton.addEventListener('click', function(event) {
            event.preventDefault();
            try { sessionStorage.removeItem(cacheKey); } catch (ignore) {}
            fetchLineup().catch(fail);
        });
    }

    setResponsesEnabled(false);
    if (studyVersion !== EXPECTED_STUDY_VERSION) {
        fail(new Error('The configured study version does not match this survey build.'));
        return;
    }
    try {
        var cached = sessionStorage.getItem(cacheKey);
        if (cached) {
            var parsed = persist(JSON.parse(cached));
            showImages(parsed).catch(fail);
            return;
        }
    } catch (ignore) {
        try { sessionStorage.removeItem(cacheKey); } catch (removeIgnore) {}
    }
    var initial = embeddedData();
    if (initial && responseProblem(initial) === '') {
        showImages(initial).catch(fail);
        return;
    }
    fetchLineup().catch(fail);
});

Qualtrics.SurveyEngine.addOnPageSubmit(function() {
    var selected = this.getSelectedChoices();
    if (selected && selected.length && this._clipRecordChoice) {
        this._clipRecordChoice(selected[0]);
    }
});

Qualtrics.SurveyEngine.addOnUnload(function() {});"""


def _question_element(qsf: dict[str, Any], qid: str) -> dict[str, Any]:
    for element in qsf["SurveyElements"]:
        if element.get("Element") == "SQ" and element.get("PrimaryAttribute") == qid:
            return element
    raise KeyError(f"Question {qid} was not found in the source QSF.")


def _element(qsf: dict[str, Any], element_type: str) -> dict[str, Any]:
    matches = [item for item in qsf["SurveyElements"] if item.get("Element") == element_type]
    if len(matches) != 1:
        raise ValueError(f"Expected one {element_type} element, found {len(matches)}.")
    return matches[0]


def _block(source_blocks: dict[str, Any], block_id: str) -> dict[str, Any]:
    for block in source_blocks.values():
        if block.get("ID") == block_id:
            return copy.deepcopy(block)
    raise KeyError(f"Block {block_id} was not found in the source QSF.")


def _flow_block(block_id: str, flow_id: str, *, default: bool = False) -> dict[str, Any]:
    return {
        "Type": "Block" if default else "Standard",
        "ID": block_id,
        "FlowID": flow_id,
        "Autofill": [],
    }


def _embedded_field(field: str, value: str, *, analyze_text: bool = False) -> dict[str, Any]:
    return {
        "Description": field,
        "Type": "Custom",
        "Field": field,
        "VariableType": "String",
        "DataVisibility": [],
        "AnalyzeText": analyze_text,
        "Value": value,
    }


def _embedded_flow(flow_id: str, fields: list[tuple[str, str, bool]]) -> dict[str, Any]:
    return {
        "Type": "EmbeddedData",
        "FlowID": flow_id,
        "EmbeddedData": [
            _embedded_field(name, value, analyze_text=analyze_text)
            for name, value, analyze_text in fields
        ],
    }


def _description_question(survey_id: str) -> dict[str, Any]:
    payload = {
        "QuestionText": VERBAL_QUESTION_HTML,
        "QuestionJS": DESCRIPTION_QUESTION_JS,
        "DefaultChoices": False,
        "DataExportTag": "VerbalDescription",
        "QuestionType": "TE",
        "Selector": "ML",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "InputWidth": 700,
            "InputHeight": 180,
        },
        "QuestionDescription": "Please describe the person you saw in the video.",
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
        "SearchSource": {"AllowFreeResponse": "false"},
        "QuestionID": VERBAL_QID,
    }
    return {
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": VERBAL_QID,
        "SecondaryAttribute": "Please describe the person you saw in the video.",
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def _lineup_question(survey_id: str) -> dict[str, Any]:
    choices: dict[str, dict[str, str]] = {}
    for position in range(1, 7):
        choices[str(position)] = {
            "Display": (
                f'<span class="clip-choice-content"><img class="clip-lineup-image" '
                f'data-position="{position}" alt="Lineup member {position}" '
                'referrerpolicy="no-referrer">'
                f'<span>Person {position}</span></span>'
            )
        }
    choices["7"] = {"Display": '<span class="clip-not-present">Not Present</span>'}
    choices["8"] = {
        "Display": (
            '<span class="clip-not-present">Technical problem &mdash; end the study '
            'and receive the completion code</span>'
        )
    }

    payload = {
        "QuestionText": (
            '<div id="clipLineupStatus" role="status" aria-live="polite">'
            'Preparing your lineup&hellip; '
            '<button type="button" id="clipLineupRetry" style="display:none">Retry</button>'
            "</div>"
        ),
        "QuestionJS": LINEUP_QUESTION_JS,
        "DefaultChoices": False,
        "DataExportTag": "LineupChoice",
        "QuestionType": "MC",
        "Selector": "SAVR",
        "SubSelector": "TX",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": "Dynamic CFD contrastive CLIP lineup response",
        "Choices": choices,
        "ChoiceOrder": list(range(1, 9)),
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 9,
        "NextAnswerId": 1,
        "QuestionID": LINEUP_QID,
    }
    return {
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": LINEUP_QID,
        "SecondaryAttribute": "Dynamic CFD contrastive CLIP lineup response",
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def _technical_failure_question(survey_id: str) -> dict[str, Any]:
    payload = {
        "QuestionText": TECHNICAL_FAILURE_HTML,
        "DefaultChoices": False,
        "DataExportTag": "TechnicalFailureEnd",
        "QuestionType": "DB",
        "Selector": "TB",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": "Technical failure debrief and completion code",
        "ChoiceOrder": [],
        "Validation": {"Settings": {"Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
        "QuestionID": TECHNICAL_QID,
    }
    return {
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": TECHNICAL_QID,
        "SecondaryAttribute": "Technical failure debrief and completion code",
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def build_qsf(source: dict[str, Any]) -> dict[str, Any]:
    qsf = copy.deepcopy(source)
    survey_id = qsf["SurveyEntry"]["SurveyID"]
    qsf["SurveyEntry"]["SurveyName"] = (
        "CFD contrastive CLIP fillers from verbal description (no groups)"
    )
    qsf["SurveyEntry"]["SurveyDescription"] = (
        "Consent, mock-crime video, verbal description, five-minute Tetris, "
        "and a six-person target-absent CFD White Male neutral-expression lineup "
        "selected with the frozen ViT-L-14 contrastive profile, followed by the original measures."
    )
    qsf["SurveyEntry"]["SurveyStatus"] = "Inactive"
    qsf["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update the ethically relevant descriptions while keeping the source survey's
    # three-checkbox confirmation structure and validation unchanged.
    text_updates = {
        "QID1": (PARTICIPANT_INFORMATION_HTML, "Participant information"),
        "QID2": (CONSENT_INFORMATION_HTML, "Consent information"),
        "QID23": (DEBRIEF_HTML, "Participant debrief"),
    }
    for qid, (html, description) in text_updates.items():
        element = _question_element(qsf, qid)
        element["SecondaryAttribute"] = description
        element["Payload"]["QuestionText"] = html
        element["Payload"]["QuestionDescription"] = description

    # Keep all three consent confirmations mandatory, but correct the grammar.
    consent = _question_element(qsf, "QID3")["Payload"]
    consent["Choices"]["1"]["Display"] = "if you agree to participate"

    # Tetris is still exactly 300 seconds. Correct its iframe title and use the
    # otherwise idle five minutes to prefetch the CLIP-selected lineup.
    tetris_info = _question_element(qsf, "QID16")["Payload"]
    tetris_info["QuestionJS"] = TETRIS_PREFETCH_JS
    tetris_frame = _question_element(qsf, "QID19")["Payload"]
    tetris_frame["QuestionText"] = tetris_frame["QuestionText"].replace(
        'title="2048 Puzzle Game"', 'title="Tetris game"'
    )

    # The new experiment has one all-filler, target-absent lineup.
    lineup_instructions = _question_element(qsf, "QID50")["Payload"]
    lineup_instructions["DataExportTag"] = "LineupInstructions_CLIP_TA"
    lineup_instructions["QuestionText"] = (
        "If you see the person from the video, select him. "
        "If you do not see him, select 'Not Present'."
    )
    lineup_instructions["QuestionDescription"] = "Dynamic lineup instructions"
    lineup_timing = _question_element(qsf, "QID44")["Payload"]
    lineup_timing["DataExportTag"] = "LineupTiming_CLIP_TA"

    keep_qids = {
        "QID1", "QID2", "QID3", "QID4", "QID5", "QID6", "QID8",
        "QID16", "QID18", "QID19", "QID20", "QID21", "QID23", "QID24",
        "QID33", "QID38", "QID44", "QID50", "QID57", "QID58",
    }
    non_questions = [
        element
        for element in qsf["SurveyElements"]
        if element.get("Element") not in {"SQ", "NT"}
    ]
    questions = [
        element
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SQ" and element.get("PrimaryAttribute") in keep_qids
    ]
    questions.extend(
        [
            _description_question(survey_id),
            _lineup_question(survey_id),
            _technical_failure_question(survey_id),
        ]
    )
    qsf["SurveyElements"] = non_questions + questions

    source_blocks = _element(qsf, "BL")["Payload"]
    block_ids = [
        "BL_0DMGxEntmozYVWC",  # participant information
        "BL_0OqDcKGpw2jWluK",  # consent
        "BL_ekOpUcXBI3IjIa2",  # before video
        "BL_5yVilrkFXEgZrYq",  # video
        "BL_ekriNg4cevKqNgi",  # Tetris
        "BL_dipvdBHqadR4it8",  # lineup instructions
        LINEUP_BLOCK_ID,        # dynamic lineup (repurposed source TA block)
        "BL_3aatmb74kFFTCtg",  # attention
        "BL_eL0bb55KBbLStsa",  # demographics
        "BL_bE2GaSKpJIKQ5o2",  # debrief
        "BL_eKFbu75iFon1lqu",  # completion/comments
    ]
    blocks: list[dict[str, Any]] = []
    for block_id in block_ids:
        block = _block(source_blocks, block_id)
        if block_id == LINEUP_BLOCK_ID:
            block["Description"] = "CFD contrastive CLIP target-absent lineup (no groups)"
            block["BlockElements"] = [
                {"Type": "Question", "QuestionID": "QID50"},
                {"Type": "Question", "QuestionID": LINEUP_QID},
                {"Type": "Question", "QuestionID": "QID38"},
                {"Type": "Question", "QuestionID": "QID44"},
            ]
        blocks.append(block)

    verbal_block = {
        "Type": "Standard",
        "SubType": "",
        "Description": "Verbal description",
        "ID": VERBAL_BLOCK_ID,
        "BlockElements": [{"Type": "Question", "QuestionID": VERBAL_QID}],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Expanded",
        },
    }
    blocks.insert(4, verbal_block)
    blocks.append(
        {
            "Type": "Standard",
            "SubType": "",
            "Description": "Technical failure debrief and completion code",
            "ID": TECHNICAL_BLOCK_ID,
            "BlockElements": [{"Type": "Question", "QuestionID": TECHNICAL_QID}],
            "Options": {
                "BlockLocking": "false",
                "RandomizeQuestions": "false",
                "BlockVisibility": "Collapsed",
            },
        }
    )
    _element(qsf, "BL")["Payload"] = {str(index): block for index, block in enumerate(blocks)}

    initial_fields: list[tuple[str, str, bool]] = [
        ("RandomSurveyID", "${rand://int/10000:99999}", False),
        (
            "lineupSeed",
            "${e://Field/ResponseID}-${rand://int/100000000:999999999}-"
            "${rand://int/100000000:999999999}-${rand://int/100000000:999999999}",
            False,
        ),
        ("CLIP_API_URL", "https://YOUR-CLIP-SERVICE.example.org/v1/lineups", False),
        ("studyVersion", STUDY_VERSION, False),
        ("expectedClipScoringContractVersion", SCORING_CONTRACT_VERSION, False),
        ("expectedClipScoringContractHash", SCORING_CONTRACT_SHA256, False),
        ("expectedClipSelectionProfileID", SELECTION_PROFILE_ID, False),
        ("expectedClipSelectionProfileHash", SELECTION_PROFILE_SHA256, False),
        ("expectedClipModel", EXPECTED_MODEL_NAME, False),
        ("expectedClipMediaManifestHash", EXPECTED_MEDIA_MANIFEST_SHA256, False),
        ("expectedClipModelSnapshotHash", EXPECTED_MODEL_SNAPSHOT_SHA256, False),
        ("lineupSize", "6", False),
        ("lineupMode", "allFillers", False),
        ("targetLineup", "targetAbsent", False),
        ("similarityType", SIMILARITY_TYPE, False),
        ("SelectionMade", "false", False),
        ("SelectedPosition", "None", False),
        ("SelectedImage", "None", False),
        ("selectedRetrievalRank", "", False),
        ("selectedRetrievalScore", "", False),
        ("selectedClipRank", "", False),
        ("selectedClipScore", "", False),
        ("Outcome", "None", False),
        ("responseType", "None", False),
        ("confidence", "0", False),
        ("FirstClickonLineupPage", "", False),
        ("lineupPreloadStarted", "false", False),
        ("lineupLoadTimeMs", "", False),
        ("LineupDecisionRTms", "", False),
        ("LineupDecisionRTs", "", False),
        ("technicalFailure", "false", False),
        ("technicalFailureReason", "", False),
        ("technicalFailureAttempts", "0", False),
        ("clipStatus", "pending", False),
        ("clipError", "", False),
        ("clipRequestID", "", False),
        ("clipModel", "", False),
        ("clipModelRevision", "", False),
        ("clipModelSnapshotHash", "", False),
        ("clipModelSnapshotMatchVerified", "", False),
        ("clipMediaManifestHash", "", False),
        ("clipIndexHash", "", False),
        ("clipCandidateEmbeddingHash", "", False),
        ("clipImageTextRevisionMatchVerified", "", False),
        ("clipQueryHash", "", False),
        ("clipQueryHashKind", "", False),
        ("clipPromptVersion", "", False),
        ("clipPromptSetHash", "", False),
        ("clipPromptCount", "", False),
        ("clipPoolVersion", "CFD3_WM_N_v2_L14", False),
        ("clipPoolHash", "", False),
        ("clipPoolProfile", "", False),
        ("clipSelectionAlgorithm", "", False),
        ("clipPoolSize", "", False),
        ("clipServiceStudyVersion", "", False),
        ("clipQueryStrategy", "", False),
        ("clipQueryChunkCount", "", False),
        ("clipQueryTokenCount", "", False),
        ("clipContextTokens", "", False),
        ("clipEffectiveQueryHash", "", False),
        ("clipScoringContractVersion", "", False),
        ("clipScoringContractHash", "", False),
        ("clipSelectionProfileID", "", False),
        ("clipSelectionProfileVersion", "", False),
        ("clipSelectionProfileHash", "", False),
        ("clipSelectionProfileStatus", "", False),
        ("clipScoringMethod", "", False),
        ("clipEffectiveMethod", "", False),
        ("clipScoreSpace", "", False),
        ("clipAggregationVersion", "", False),
        ("clipScorerVersion", "", False),
        ("clipPlanStatus", "", False),
        ("clipContrastPlanHash", "", False),
        ("clipContrastSchemaName", "", False),
        ("clipContrastSchemaVersion", "", False),
        ("clipContrastSchemaHash", "", False),
        ("clipParserVersion", "", False),
        ("clipPromptTemplateVersion", "", False),
        ("clipPromptTemplateHash", "", False),
        ("clipContrastPairCount", "", False),
        ("clipContrastDimensionCount", "", False),
        ("clipAcceptedRuleIDs", "", False),
        ("clipAcceptedDimensions", "", False),
        ("clipAbstentionCount", "", False),
        ("clipAbstentionReasonCodes", "", False),
        ("clipFallbackUsed", "", False),
        ("clipFallbackReason", "", False),
        ("clipStandardiseComponents", "", False),
        ("clipFilterGender", "male", False),
        ("clipFilterRace", "White", False),
    ]
    for position in range(1, 7):
        initial_fields.extend(
            [
                (f"lineup{position}", "", False),
                (f"lineup{position}Url", "", False),
                (f"lineup{position}Rank", "", False),
                (f"lineup{position}Score", "", False),
            ]
        )

    attention_branch = copy.deepcopy(
        next(
            item
            for item in _element(source, "FL")["Payload"]["Flow"]
            if item.get("Type") == "Branch"
            and "QID33" in json.dumps(item.get("BranchLogic", {}), sort_keys=True)
        )
    )
    technical_failure_branch = {
        "Type": "Branch",
        "FlowID": "FL_201",
        "Description": "CLIP or image technical failure",
        "BranchLogic": {
            "0": {
                "0": {
                    "LogicType": "Question",
                    "QuestionID": LINEUP_QID,
                    "QuestionIsInLoop": "no",
                    "ChoiceLocator": f"q://{LINEUP_QID}/SelectableChoice/8",
                    "Operator": "Selected",
                    "QuestionIDFromLocator": LINEUP_QID,
                    "LeftOperand": f"q://{LINEUP_QID}/SelectableChoice/8",
                    "Type": "Expression",
                    "Description": (
                        '<span class="ConjDesc">If</span> '
                        '<span class="QuestionDesc">Dynamic CFD CLIP lineup response</span> '
                        '<span class="LeftOpDesc">Technical problem</span> '
                        '<span class="OpDesc">Is Selected</span>'
                    ),
                },
                "Type": "If",
            },
            "Type": "BooleanExpression",
        },
        "Flow": [
            _flow_block(TECHNICAL_BLOCK_ID, "FL_202"),
            {"Type": "EndSurvey", "FlowID": "FL_203"},
        ],
    }
    flow = [
        _embedded_flow("FL_200", initial_fields),
        _flow_block("BL_0DMGxEntmozYVWC", "FL_2", default=True),
        _flow_block("BL_0OqDcKGpw2jWluK", "FL_3"),
        _flow_block("BL_ekOpUcXBI3IjIa2", "FL_5"),
        _flow_block("BL_5yVilrkFXEgZrYq", "FL_24"),
        _flow_block(VERBAL_BLOCK_ID, "FL_109"),
        _embedded_flow(
            "FL_110",
            [("verbalDescription", "${q://QID59/ChoiceTextEntryValue}", True)],
        ),
        _flow_block("BL_ekriNg4cevKqNgi", "FL_6"),
        _flow_block("BL_dipvdBHqadR4it8", "FL_111"),
        _flow_block(LINEUP_BLOCK_ID, "FL_112"),
        _embedded_flow(
            "FL_113",
            [
                ("FirstClickonLineupPage", "${q://QID44/ChoiceNumericEntryValue/1}", False),
                ("confidence", "${q://QID38/ChoiceGroup/SelectedChoices}", False),
            ],
        ),
        technical_failure_branch,
        _flow_block("BL_3aatmb74kFFTCtg", "FL_51"),
        attention_branch,
        _flow_block("BL_eL0bb55KBbLStsa", "FL_4"),
        _flow_block("BL_bE2GaSKpJIKQ5o2", "FL_23"),
        _flow_block("BL_eKFbu75iFon1lqu", "FL_73"),
    ]
    flow_element = _element(qsf, "FL")
    flow_element["Payload"] = {
        "Type": "Root",
        "FlowID": "FL_1",
        "Flow": flow,
        "Properties": {"Count": 203},
    }

    options = _element(qsf, "SO")["Payload"]
    options["AnonymizeResponse"] = "Yes"
    options["CollectGeoLocation"] = "false"
    options["SurveyTitle"] = "CFD contrastive CLIP filler selection from a verbal description"
    options["SurveyMetaDescription"] = "Eyewitness verbal-description and lineup study"

    question_count = _element(qsf, "QC")
    question_count["SecondaryAttribute"] = "61"
    return qsf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the no-groups CFD/CLIP Qualtrics QSF from the supplied source survey."
    )
    parser.add_argument("--source", type=Path, required=True, help="Source QSF; the included frozen QSF can be rebuilt unchanged.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.source.open("r", encoding="utf-8-sig") as handle:
        source = json.load(handle)
    result = build_qsf(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
