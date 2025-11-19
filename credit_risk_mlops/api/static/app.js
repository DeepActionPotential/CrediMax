document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    let json = {};

    formData.forEach((value, key) => {
        if (value !== "") json[key] = isNaN(value) ? value : Number(value);
    });

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "⏳ Running prediction...";

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(json)
    });

    const data = await res.json();

    if (res.status !== 200) {
        resultDiv.innerHTML = `<span style="color:red;">Error: ${data.detail}</span>`;
        return;
    }

    resultDiv.innerHTML = `
        <b>Probability of Default:</b> ${data.probability.toFixed(4)}<br>
        <b>Prediction:</b> ${
            data.prediction === 1 ? " High Risk" : " Low Risk"
        }
    `;
});
