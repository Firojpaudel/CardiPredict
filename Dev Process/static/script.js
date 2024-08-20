document.getElementById('uploadForm').onsubmit = async function (event) {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput').files[0];
    formData.append('file', fileInput);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        document.getElementById('result').innerText = result.prediction !== undefined 
            ? result.prediction 
            : 'Error: ' + result.error;
    } catch (error) {
        document.getElementById('result').innerText = 'Error: ' + error.message;
    }
}
