function runSimulation() {
    // Get input values from the form
    var z = document.getElementById("z").value;
    
    // Add more variables as needed

    // Send input to the Python script using XHR or Fetch API
    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            z: z,
            // Include other parameters as needed
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Display the simulation result on the page
        var resultDiv = document.getElementById("simulationResult");
        resultDiv.innerHTML = "<p>Simulation Result: </p>";

        var image = document.getElementById("simulationImage");
    
        image = document.createElement("img");
        image.id = "simulationImage";
        image.src = "static/simulation_plot.png?random="+new Date().getTime();
        resultDiv.appendChild(image);
    }) 
    .catch((error) => {
        console.error('Error:', error);
    });
}
