let number = document.getElementById('number');
let context_number = number.getContext('2d', { willReadFrequently: true });
let result = document.getElementById('result');
let context_result = result.getContext('2d', { willReadFrequently: true });

let isDrawing = false;
let isErasing = false;
let matrix = [];

clearCanvas();

number.addEventListener('mousedown', startDrawing);
number.addEventListener('mousemove', draw);
number.addEventListener('mouseup', stopDrawing);
number.addEventListener('mouseout', stopDrawing);

function clearCanvas() {
    context_number.fillStyle = 'black';
    context_number.fillRect(0, 0, number.width, number.height);
    context_result.fillStyle = 'black';
    context_result.fillRect(0, 0, result.width, result.height);
}

// Drawing functions, including start and stop drawing

function startDrawing(e) {
    isDrawing = true;

    let x = Math.floor((e.clientX - number.getBoundingClientRect().left));
    let y = Math.floor((e.clientY - number.getBoundingClientRect().top));

    
    if (isErasing){context_number.strokeStyle = 'black';}
    else{context_number.strokeStyle = 'white';}
    context_number.lineWidth = 10;
    context_number.lineCap = 'round';

    context_number.beginPath();
    context_number.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;

    if (isErasing){context_number.strokeStyle = 'black';}
    else{context_number.strokeStyle = 'white';}

    let x = Math.floor((e.clientX - number.getBoundingClientRect().left));
    let y = Math.floor((e.clientY - number.getBoundingClientRect().top));
    context_number.lineTo(x, y);
    context_number.stroke();
    convertCanvasToMatrix(context_number);
}

function stopDrawing() {
    isDrawing = false;
    Recognize();
}

function toggleEraser() {
    isErasing = !isErasing;
    if (isErasing){document.querySelector('#Eraser').textContent = 'Eraser: on';}
    else{document.querySelector('#Eraser').textContent = 'Eraser: off';}
}

function getMax(a){
    return Math.max(...a.map(e => Array.isArray(e) ? getMax(e) : e));
  }

function convertCanvasToMatrix(context) {
    const matrixSize = 28;
    const blockSize = 280 / matrixSize;
    matrix = [];

    for (let i = 0; i < matrixSize; i++) {
        let row = [];
        for (let j = 0; j < matrixSize; j++) {
            let imageData = context.getImageData(j * blockSize, i * blockSize, blockSize, blockSize);
            let totalBrightness = 0;
            for (let k = 0; k < imageData.data.length; k += 4) {
                // Calculate the brightness using the luminance formula
                let brightness = 0.299 * imageData.data[k] + 0.587 * imageData.data[k + 1] + 0.114 * imageData.data[k + 2];
                totalBrightness += brightness;
            }
            // Average brightness for the block, normalized to 0-1 range
            row.push(totalBrightness / (blockSize * blockSize * 255));
        }
        matrix.push(row);
    }
}

// Simulation
function Recognize() {
    // Send the parameters to server
    fetch('/recognize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            matrix: matrix,
        }),
    })
    // Reveive response
    .then(response => response.json())
    // Plot on the canvas
    .then(data => {
        plot = data.plot;
        prediction = data.prediction;
        ratio = 255 / getMax(plot)

        for (let i = 0; i < plot.length; i++) {
            for (let j = 0; j < plot.length; j++){
                let pixelValue = plot[j][i] * ratio;
                context_result.fillStyle = `rgba(${pixelValue}, ${pixelValue}, ${pixelValue}, 1)`;
                context_result.fillRect(i, j, 1, 1);
            }
        }
        document.getElementById("prediction").textContent = "The number is: " + prediction;

    }) 
    // Log error message
    .catch((error) => {
        console.error('Error:', error);
    });
}

