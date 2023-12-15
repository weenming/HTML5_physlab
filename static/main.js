let canvas = document.getElementById('drawCanvas');
let context = canvas.getContext('2d', { willReadFrequently: true });
let canvas_r = document.getElementById('resultCanvas');
let context_r = canvas_r.getContext('2d', { willReadFrequently: true });
let canvas_p = document.getElementById('plotCanvas');
let context_p = canvas_p.getContext('2d', { willReadFrequently: true });

let isDrawing = false;
let isErasing = false;
let isMeasuring = true;
let showGrid = false;

let lineEnds = [];
let points = [];
let plot = [];
let clickCount = 0;

preset();

// Preset
function preset(){
    clearCanvas();
    var types = document.getElementById("types").value
    context.fillStyle = 'white';
    if (types === "Double slit"){
        context.fillRect(0, canvas.height/2 - 11, canvas.width, 2)
        context.fillRect(0, canvas.height/2 + 9, canvas.width, 2)
    }
    if (types === "Circle"){
        radius = 20;
        centerX = canvas.width / 2;
        centerY = canvas.height / 2;
        context.beginPath();
        context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
        context.fill();
    }
    if (types === "Square"){
        context.fillRect(canvas.width/2-10, canvas.height/2-10, 20,20)
    }
    runSimulation();
}

// Event listeners for drawing input
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Drawing functions, including start and stop drawing

function startDrawing(e) {
    isDrawing = true;

    let x = Math.floor((e.clientX - canvas.getBoundingClientRect().left));
    let y = Math.floor((e.clientY - canvas.getBoundingClientRect().top));

    
    if (isErasing){context.strokeStyle = 'black';}
    else{context.strokeStyle = 'white';}
    let BrushSize = document.getElementById("brushSlider").value;
    context.lineWidth = BrushSize;
    context.lineCap = 'square';

    context.beginPath();
    context.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;

    if (isErasing){context.strokeStyle = 'black';}
    else{context.strokeStyle = 'white';}

    let x = Math.floor((e.clientX - canvas.getBoundingClientRect().left));
    let y = Math.floor((e.clientY - canvas.getBoundingClientRect().top));
    context.lineTo(x, y);
    context.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

// Canvas utilities, including eraser, brush size and canvas size.
function clearCanvas() {
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context_r.fillStyle = 'black';
    context_r.fillRect(0, 0, canvas_r.width, canvas_r.height);
    context_p.fillStyle = 'black';
    context_p.fillRect(0, 0, canvas_p.width, canvas_p.height);
}

function Grid(){
    if (showGrid){
        let d = canvas.width/2
        context.fillStyle = 'white';
        context_r.fillStyle = 'white';
        for (let i = 0; i<d; i+=50){
            //context.fillRect(0,d+i,canvas.width,0.5);
            context_r.fillRect(0,d+i,canvas.width,0.5);
            //context.fillRect(0,d-i,canvas.width,0.5);
            context_r.fillRect(0,d-i,canvas.width,0.5);
            //context.fillRect(d+i,0,0.5,canvas.width);
            context_r.fillRect(d+i,0,0.5,canvas.width);
            //context.fillRect(d-i,0,0.5,canvas.width);
            context_r.fillRect(d-i,0,0.5,canvas.width);
        }
    }
}

function toggleEraser() {
    isErasing = !isErasing;
    if (isErasing){document.querySelector('#Eraser').textContent = 'Eraser: on';}
    else{document.querySelector('#Eraser').textContent = 'Eraser: off';}
}

function toggleGrid() {
    showGrid = !showGrid;
    runSimulation();
    if (showGrid){document.querySelector('#showGrid').textContent = 'Show Grid: on';}
    else{document.querySelector('#showGrid').textContent = 'Show Grid: off';}
}

function changeBrushSize(){
    let BrushSize = document.getElementById("brushSlider").value;
    document.getElementById('brushValue').textContent = "Brush width:"+BrushSize;
}

function changeCanvasSize() {
    let newSideLength = document.getElementById('canvasSlider').value;

    canvas.width = newSideLength;
    canvas.height = newSideLength;
    canvas_r.width = newSideLength;
    canvas_r.height = newSideLength;
    canvas_p.width = newSideLength*2 +10;
    canvas_p.height = newSideLength;

    clearCanvas();

    document.getElementById('sliderValue').textContent = "Canvas width:"+newSideLength;
}

// Event listener for measurement
canvas_r.addEventListener('click', function(e) {
    if (isMeasuring){
        clickCount++;
    
        let x = Math.floor(e.clientX - canvas_r.getBoundingClientRect().left);
        let y = Math.floor(e.clientY - canvas_r.getBoundingClientRect().top);
        
        context_r.fillStyle = 'yellow';
        context_r.fillRect(x, y, 1, 1);

        if (clickCount === 3){
            clickCount = 1;
            lineEnds = [];
            context_p.fillStyle = 'black';
            context_p.fillRect(0, 0, canvas_p.width, canvas_p.height);
            runSimulation();
            context_r.fillStyle = 'yellow';
            context_r.fillRect(x, y, 1, 1);
        }
        
        lineEnds.push({ x: x, y: y });
        
        if (clickCount === 2) {
            runSimulation();
        }
    }
});

// Measurement functions
function toggleMeasurement() {
    isMeasuring = !isMeasuring;
    if (isMeasuring){document.querySelector('#Measurement').textContent = 'Measurement: on';}
    else{document.querySelector('#Measurement').textContent = 'Measurement: off';}
}

function drawLine(ends) {
    if (ends.length < 2) {
        return;
    }

    context_r.beginPath();
    context_r.moveTo(ends[0].x, ends[0].y);
    context_r.lineTo(ends[1].x, ends[1].y);
    context_r.strokeStyle = 'yellow';
    context_r.stroke();
}

function getPoints(ends) {
    if (ends.length < 2) {
        return;
    }

    points = [];
    plot = [];
    x0 = ends[0].x
    y0 = ends[0].y
    x1 = ends[1].x
    y1 = ends[1].y
    let dx = Math.abs(x1 - x0);
    let dy = Math.abs(y1 - y0);
    let sx = (x0 < x1) ? 1 : -1;
    let sy = (y0 < y1) ? 1 : -1;
    let err = dx - dy;

    while (true) {
        points.push({ x: x0, y: y0 });

        if ((x0 === x1) && (y0 === y1)) break;
        let e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

var dotMul=function(v1,v2){ return   v1.x*v2.x+v1.y*v2.y;  }
var substract = function(v1, v2){return {x: v1.x-v2.x, y: v1.y-v2.y};}

function createPlot(ends, points, result, y_ratio){
    if (ends.length < 2) {
        return;
    }
    x0 = ends[0].x
    y0 = ends[0].y
    x1 = ends[1].x
    y1 = ends[1].y
    let dx = Math.abs(x1 - x0);
    let dy = Math.abs(y1 - y0);
    let len = Math.sqrt(dx*dx+dy*dy);
    let e = {x: (x1-x0)/len, y: (y1-y0)/len};
    for (let i = 0; i < points.length; i++){
        plot.push({x : Math.abs(dotMul(e, substract(points[i],ends[0]))), y : result[points[i].y][points[i].x]});
    }
    console.log(plot)
    context_p.fillStyle = 'black';
    context_p.fillRect(0, 0, canvas_p.width, canvas_p.height);
    let x_ratio = canvas_p.width/plot[plot.length-1].x;
    context_p.strokeStyle = 'yellow';
    context_p.beginPath();
    context_p.moveTo(plot[0].x * x_ratio, canvas_p.height - plot[0].y * y_ratio);
    for (let i = 1; i < plot.length; i++){
        context_p.lineTo(plot[i].x * x_ratio, canvas_p.height - plot[i].y * y_ratio)
    }
    context_p.stroke();
}

// Save measurement
function convertToCSV(resultData) {
    let csvContent = "";

    // Add CSV header if needed
    csvContent += "x,y\r\n";

    resultData.forEach(function (item) {
        let row = [item.x, item.y];
        let rowString = row.join(",");
        csvContent += rowString + "\r\n";
    });

    return csvContent;
}

document.getElementById("saveResultButton").addEventListener("click", function () {
    // Generate the CSV content
    let csvData = convertToCSV(plot);

    // Create a blob with the CSV data
    let blob = new Blob([csvData], { type: "text/csv;charset=utf-8;" });

    // Create a download link
    let link = document.createElement("a");
    if (link.download !== undefined) {
        let url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "result.csv");
        link.style.visibility = "hidden";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } else {
        alert("Your browser does not support the download feature. Please try a different browser.");
    }
});


function getMax(a){
    return Math.max(...a.map(e => Array.isArray(e) ? getMax(e) : e));
  }

// Simulation
function runSimulation() {
    // Process all the parameters from the page
    var z = document.getElementById("z").value;
    var wl = document.getElementById("wl").value;
    var grid = document.getElementById("grid").value;
    var types = document.getElementById("types").value;
    let imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let matrix = [];
    for (let i = 0; i < canvas.height; i++) {
        let row = [];
        for (let j = 0; j < canvas.width; j++) {
            let index = ((i * canvas.width) + j) * 4;
            let isBlack = imageData[index] === 0;
            row.push(isBlack ? 0 : 1);
        }
        matrix.push(row);
    }
    // Send the parameters to server
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
    // Reveive response
    .then(response => response.json())
    // Plot on the canvas
    .then(data => {
        result = data.result
        ratio = 255 / getMax(result)
        

        for (let i = 0; i < canvas_r.width; i++) {
            for (let j = 0; j < canvas_r.width; j++){
                let pixelValue = result[j][i] * ratio;
                context_r.fillStyle = `rgba(${pixelValue}, ${pixelValue}, ${pixelValue}, 1)`;
                context_r.fillRect(i, j, 1, 1);
            }
        }

        // Display measure line
        drawLine(lineEnds);

        // Get pixels on the line
        getPoints(lineEnds);

        // Plot on the canvas_p
        let newSideLength = document.getElementById('canvasSlider').value;
        ratio = newSideLength/getMax(result)
        createPlot(lineEnds, points, result, ratio);

        Grid();

    }) 
    // Log error message
    .catch((error) => {
        console.error('Error:', error);
    });
}

