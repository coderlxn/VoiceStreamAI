/**
 * VoiceStreamAI Client - WebSocket-based real-time transcription
 *
 * Contributor:
 * - Alessandro Saccoia - alessandro.saccoia@gmail.com
 */

let websocket;
let context;
let processor;
let globalStream;
// 创建一个新的语音合成对象
let SST ;

const websocket_uri = 'ws://localhost:6006';
const bufferSize = 4096;
let isRecording = false;

function initWebSocket() {
    const websocketAddress = document.getElementById('websocketAddress').value;
    chunk_length_seconds = document.getElementById('chunk_length_seconds').value;
    chunk_offset_seconds = document.getElementById('chunk_offset_seconds').value;
    const selectedLanguage = document.getElementById('languageSelect').value;
    language = selectedLanguage !== 'multilingual' ? selectedLanguage : null;
    
    if (!websocketAddress) {
        console.log("WebSocket address is required.");
        return;
    }

    websocket = new WebSocket(websocketAddress);
    websocket.onopen = () => {
        console.log("WebSocket connection established");
        document.getElementById("webSocketStatus").textContent = 'Connected';
        document.getElementById('startButton').disabled = false;
        SST = new SpeechSynthesisUtterance();
    };
    websocket.onclose = event => {
        console.log("WebSocket connection closed", event);
        document.getElementById("webSocketStatus").textContent = 'Not Connected';
        document.getElementById('startButton').disabled = true;
        document.getElementById('stopButton').disabled = true;
    };
    websocket.onmessage = event => {
        console.log("Message from server:", event.data);
        const transcript_data = JSON.parse(event.data);
        updateTranscription(transcript_data);
    };
}

function updateTranscription(transcript_data) {
    const transcriptionDiv = document.getElementById('transcription');
    const languageDiv = document.getElementById('detected_language');

    if (transcript_data['ai_resp'] && transcript_data['ai_resp'].length > 0) {
        var ai_resp = transcript_data['ai_resp'];
        const span = document.createElement('span');
        span.textContent = '【AI Resp】: ' + ai_resp ;
        span.style.color = 'red';
        transcriptionDiv.appendChild(span);
        let speechInstance = new SpeechSynthesisUtterance(ai_resp);
        speechSynthesis.speak(speechInstance);

    } else if (transcript_data['words'] && transcript_data['words'].length > 0) {
        // Append words with color based on their probability
        transcript_data['words'].forEach(wordData => {
            const span = document.createElement('span');
            const probability = wordData['probability'];
            span.textContent = wordData['word'] + ' ';

            // Set the color based on the probability
            if (probability > 0.9) {
                span.style.color = 'green';
            } else if (probability > 0.6) {
                span.style.color = 'orange';
            } else {
                span.style.color = 'blue';
            }

            transcriptionDiv.appendChild(span);
        });

        // Add a new line at the end
        // transcriptionDiv.appendChild(document.createElement('br'));
    } else {
        // Fallback to plain text
        transcriptionDiv.textContent += transcript_data['text'] + '\n';
    }
    // Add a new line at the end
    transcriptionDiv.appendChild(document.createElement('br'));

    // Update the language information
    if (transcript_data['language'] && transcript_data['language_probability']) {
        languageDiv.textContent = transcript_data['language'] + ' (' + transcript_data['language_probability'].toFixed(2) + ')';
    }

    // Update the processing time, if available
    const processingTimeDiv = document.getElementById('processing_time');
    if (transcript_data['processing_time']) {
        processingTimeDiv.textContent = 'Processing time: ' + transcript_data['processing_time'].toFixed(2) + ' seconds';
    }
}


function startRecording() {
    if (isRecording) return;
    isRecording = true;

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    context = new AudioContext();

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        globalStream = stream;
        const input = context.createMediaStreamSource(stream);
        processor = context.createScriptProcessor(bufferSize, 1, 1);
        processor.onaudioprocess = e => processAudio(e);
        input.connect(processor);
        processor.connect(context.destination);

        sendAudioConfig();
    }).catch(error => console.error('Error accessing microphone', error));

    // Disable start button and enable stop button
    document.getElementById('startButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    if (globalStream) {
        globalStream.getTracks().forEach(track => track.stop());
    }
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (context) {
        context.close().then(() => context = null);
    }
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
}

function sendAudioConfig() {
    let selectedStrategy = document.getElementById('bufferingStrategySelect').value;
    let processingArgs = {};

    if (selectedStrategy === 'silence_at_end_of_chunk') {
        processingArgs = {
            chunk_length_seconds: parseFloat(document.getElementById('chunk_length_seconds').value),
            chunk_offset_seconds: parseFloat(document.getElementById('chunk_offset_seconds').value)
        };
    }

    const audioConfig = {
        type: 'config',
        data: {
            sampleRate: context.sampleRate,
            bufferSize: bufferSize,
            channels: 1, // Assuming mono channel
            language: language,
            processing_strategy: selectedStrategy, 
            processing_args: processingArgs
        }
    };

    websocket.send(JSON.stringify(audioConfig));
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) {
        return buffer;
    }
    var sampleRateRatio = inputSampleRate / outputSampleRate;
    var newLength = Math.round(buffer.length / sampleRateRatio);
    var result = new Float32Array(newLength);
    var offsetResult = 0;
    var offsetBuffer = 0;
    while (offsetResult < result.length) {
        var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        var accum = 0, count = 0;
        for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

function processAudio(e) {
    const inputSampleRate = context.sampleRate;
    const outputSampleRate = 16000; // Target sample rate

    const left = e.inputBuffer.getChannelData(0);
    const downsampledBuffer = downsampleBuffer(left, inputSampleRate, outputSampleRate);
    const audioData = convertFloat32ToInt16(downsampledBuffer);
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(audioData);
    }
}

function convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
        buf[l] = Math.min(1, buffer[l]) * 0x7FFF;
    }
    return buf.buffer;
}

// Initialize WebSocket on page load
//  window.onload = initWebSocket;

function toggleBufferingStrategyPanel() {
    var selectedStrategy = document.getElementById('bufferingStrategySelect').value;
    if (selectedStrategy === 'silence_at_end_of_chunk') {
        var panel = document.getElementById('silence_at_end_of_chunk_options_panel');
        panel.classList.remove('hidden');
    } else {
        var panel = document.getElementById('silence_at_end_of_chunk_options_panel');
        panel.classList.add('hidden');
    }
}
let voiceIndex = 156;
const TextToSpeaker = (text) => {
    // 创建一个新的 SpeechSynthesisUtterance 实例
    var utterance = new SpeechSynthesisUtterance(text);

    // 设置指定的语音角色
    // const xiaoxiao = getSpeaker("Microsoft Xiaoxiao Online (Natural) - Chinese (Mainland)");
    const putong = getSpeaker("Google 普通话（中国大陆）");
    // if (xiaoxiao) {
    //   utterance.voice = xiaoxiao;
    // } else if (putong) {
    //   utterance.voice = putong;
    // }
    const speaker = getSpeakerAt(voiceIndex);
    console.log('USE voice:' + voiceIndex + " > " + speaker)
    voiceIndex ++;
    if(speaker){
        utterance.voice = speaker
    }else {
        console.log('USE voice:' + voiceIndex + " > Google 普通话")
        utterance.voice = putong
    }

    // 可以设置朗读的属性，如语速和音调
    utterance.rate = 1; // 语速，可以是0.1到10之间的值，默认为1
    utterance.pitch = 1; // 音调，可以是0到2之间的值，默认为1

    // 使用 SpeechSynthesis 接口的 speak 方法开始朗读
    window.speechSynthesis.speak(utterance);
  };

const getSpeaker = (speakName) => {
    const sp = window.speechSynthesis.getVoices().find((voice) => voice.name === speakName);
    if (sp) {
      return sp;
    } else {
      return null;
    }
  };


const getSpeakerAt = (index) => {
    if(index >= 0 && index < window.speechSynthesis.getVoices().length){
        return window.speechSynthesis.getVoices().at(index);
    }else{
        return window.speechSynthesis.getVoices().at(0);
    }
  };