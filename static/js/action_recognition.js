let model;
let webcam;

async function setupWebcam() {
    const video = document.getElementById('webcam');
    const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModel() {
    model = await posenet.load();
}

async function detectPose() {
    const pose = await model.estimateSinglePose(webcam);
    const action = interpretPose(pose);
    document.getElementById('action').textContent = action;
}

function interpretPose(pose) {
    // This is a simplified interpretation. In a real application,
    // you'd use more sophisticated logic to determine the action.
    const leftWrist = pose.keypoints.find(k => k.part === 'leftWrist');
    const rightWrist = pose.keypoints.find(k => k.part === 'rightWrist');

    if (leftWrist.position.y < pose.keypoints[0].position.y &&
        rightWrist.position.y < pose.keypoints[0].position.y) {
        return "Hands raised";
    } else {
        return "Standing";
    }
}

async function init() {
    webcam = await setupWebcam();
    await loadModel();
    setInterval(detectPose, 100);
}

init();