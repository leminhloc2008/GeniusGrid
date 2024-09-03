document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            appendMessage('You', message);
            userInput.value = '';
            const response = await getBotResponse(message);
            appendMessage('Bot', response);
        }
    });

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function getBotResponse(message) {
        // In a real application, this would make an API call to a backend service
        // For now, we'll just return a simple response
        return "I'm sorry, I'm just a simple bot. I don't understand complex queries yet.";
    }
});