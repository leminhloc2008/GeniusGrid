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
    model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
    );
}