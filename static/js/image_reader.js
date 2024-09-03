document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const imageContainer = document.getElementById('image-container');
    const readImageButton = document.getElementById('read-image');
    const imageText = document.getElementById('image-text');

    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            const img = document.createElement('img');
            img.src = event.target.result;
            img.className = 'max-w-full h-auto';
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
        };

        reader.readAsDataURL(file);
    });

    readImageButton.addEventListener('click', () => {
        const img = imageContainer.querySelector('img');
        if (img) {
            Tesseract.recognize(img.src)
                .then(({ data: { text } }) => {
                    imageText.textContent = text;
                })
                .catch(error => {
                    console.error(error);
                    imageText.textContent = 'Error reading image';
                });
        } else {
            imageText.textContent = 'Please upload an image first';
        }
    });
});