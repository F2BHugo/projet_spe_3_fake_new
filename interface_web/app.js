async function analyze() {
  const text = document.getElementById("inputText").value;
  const author = document.getElementById("author").value;
  const orientation = document.getElementById("orientation").value;

  const resultDiv = document.getElementById("result");
  const predictionP = document.getElementById("prediction");
  const confidenceP = document.getElementById("confidence");

  if (text.trim() === "") {
    alert("Veuillez entrer une déclaration.");
    return;
  }

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text,
        author: author,
        orientation: orientation
      })
    });

    const data = await response.json();

    resultDiv.classList.remove("hidden");

    const labelText =
      data.label === 1 ? "✅ Plutôt VRAI" : "❌ Plutôt FAUX";

    predictionP.innerText = "Verdict : " + labelText;
    confidenceP.innerText =
      "Confiance du modèle : " + (data.confidence * 100).toFixed(1) + "%";

  } catch (error) {
    console.error(error);
    alert("Erreur lors de l'analyse.");
  }
}