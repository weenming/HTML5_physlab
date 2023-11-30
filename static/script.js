function runSimulation() {
    var z = document.getElementById("z").value;
    var wl = document.getElementById("wl").value;
    var grid = document.getElementById("grid").value;
    var types = document.getElementById("types").value;

    let imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let matrix = [];

    for (let i = 0; i < canvas.height / 2; i++) {
        let row = [];
        for (let j = 0; j < canvas.width / 2; j++) {
            let index = ((i * canvas.width) + j) * 8;
            let isBlack = imageData[index] === 0;
            row.push(isBlack ? 0 : 1);
        }
        matrix.push(row);
    }

    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            z: z,
            wl: wl,
            grid: grid,
            matrix: matrix,
            types: types,
        }),
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv0 = document.getElementById("InitialCondition");
        resultDiv0.innerHTML = "<p>Initial Condition: </p>";

        var image0 = document.getElementById("startImage");
        image0 = document.createElement("img");
        image0.id = "startImage";
        image0.src = "static/start_plot.png?random="+new Date().getTime();
        resultDiv0.appendChild(image0);

        var resultDiv1 = document.getElementById("simulationResult");
        resultDiv1.innerHTML = "<p>Simulation : </p>";

        var image1 = document.getElementById("simulationImage");
        image1 = document.createElement("img");
        image1.id = "simulationImage";
        image1.src = "static/simulation_plot.png?random="+new Date().getTime();
        resultDiv1.appendChild(image1);
    }) 
    .catch((error) => {
        console.error('Error:', error);
    });
}

let canvas = document.getElementById('drawCanvas');
let context = canvas.getContext('2d');
let isDrawing = false;
let isErasing = false;

context.fillStyle = 'black';
context.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;

    let x = Math.floor((e.clientX - canvas.getBoundingClientRect().left) / 2);
    let y = Math.floor((e.clientY - canvas.getBoundingClientRect().top) / 2);

    
    if (isErasing){context.strokeStyle = 'black';}
    else{context.strokeStyle = 'white';}
    context.lineWidth = 2;
    context.lineCap = 'round';

    context.beginPath();
    context.moveTo(x * 2, y * 2);
}

function draw(e) {
    if (!isDrawing) return;

    //context.fillStyle = 'white';
    if (isErasing){context.strokeStyle = 'black';}
    else{context.strokeStyle = 'white';}
    let x = Math.floor((e.clientX - canvas.getBoundingClientRect().left) / 2);
    let y = Math.floor((e.clientY - canvas.getBoundingClientRect().top) / 2);
    context.lineTo(x * 2, y * 2);
    context.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
}

function toggleEraser() {
    isErasing = !isErasing;
}