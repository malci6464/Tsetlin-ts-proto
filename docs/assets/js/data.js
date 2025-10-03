async function fetchJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

async function fetchCSV(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  const text = await response.text();
  return parseCSV(text);
}

function parseCSV(text) {
  const [headerLine, ...rows] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return rows.filter(Boolean).map((row) => {
    const values = row.split(",");
    return headers.reduce((acc, key, idx) => {
      acc[key] = values[idx];
      return acc;
    }, {});
  });
}

function createSummaryList(target, entries) {
  target.innerHTML = "";
  const fragment = document.createDocumentFragment();
  entries.forEach(([label, value]) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${label}:</strong> ${value}`;
    fragment.appendChild(li);
  });
  target.appendChild(fragment);
}

function createTable(target, rows, columns) {
  target.innerHTML = "";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(({ key, label }) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach(({ key, render }) => {
      const td = document.createElement("td");
      const value = typeof render === "function" ? render(row[key], row) : row[key];
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  target.appendChild(table);
}

async function loadMaritimeDataset() {
  const basePath = "time_series_outputs/maritime/20251003_125026";
  const summary = await fetchJSON(`${basePath}/run_summary.json`);
  const accuracyHistory = await fetchCSV(`${basePath}/accuracy_history.csv`);
  const sampleWindows = await fetchCSV(`${basePath}/sample_windows.csv`);
  const testPredictions = await fetchCSV(`${basePath}/test_predictions.csv`);

  populateSummary(summary);
  renderAccuracyChart(accuracyHistory);
  renderSampleWindows(sampleWindows.slice(0, 20));
  renderPredictionSummary(testPredictions);
}

function populateSummary(summary) {
  const metadata = summary.metadata || {};
  const hyperparameters = summary.hyperparameters || {};

  const metadataList = document.querySelector("#metadata-list");
  if (metadataList) {
    createSummaryList(metadataList, [
      ["Scenario", summary.scenario ?? "maritime"],
      ["Sensor count", metadata.sensors?.length ?? 0],
      ["Window size", metadata.window_size],
      ["Noise probability", metadata.noise_probability],
      ["Event probability", metadata.event_probability],
      ["Sequences", metadata.n_sequences],
      ["Sequence length", metadata.sequence_length],
      ["Total samples", metadata.samples],
      ["Positive windows", metadata.positives],
    ]);
  }

  const sensorList = document.querySelector("#sensor-list");
  if (sensorList && Array.isArray(metadata.sensors)) {
    sensorList.innerHTML = metadata.sensors.map((sensor) => `<li>${sensor}</li>`).join("");
  }

  const hyperparameterList = document.querySelector("#hyperparameter-list");
  if (hyperparameterList) {
    createSummaryList(hyperparameterList, Object.entries(hyperparameters).map(([key, value]) => [key, value]));
  }

  const accuracyBadges = document.querySelector("#accuracy-badges");
  if (accuracyBadges) {
    const { training } = summary;
    const finalAcc = training?.final_test_accuracy;
    const bestAcc = training?.best_test_accuracy;
    accuracyBadges.innerHTML = `
      <span class="badge">Final test accuracy: ${(finalAcc * 100).toFixed(2)}%</span>
      <span class="badge">Best test accuracy: ${(bestAcc * 100).toFixed(2)}%</span>
    `;
  }
}

function renderAccuracyChart(accuracyHistory) {
  const ctx = document.getElementById("accuracy-chart");
  if (!ctx) return;
  const epochs = accuracyHistory.map((row) => Number(row.epoch));
  const trainAcc = accuracyHistory.map((row) => Number(row.train_accuracy));
  const testAcc = accuracyHistory.map((row) => Number(row.test_accuracy));

  new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train accuracy",
          data: trainAcc,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.15)",
          tension: 0.25,
          fill: true,
        },
        {
          label: "Test accuracy",
          data: testAcc,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.15)",
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        y: {
          min: 0.85,
          max: 1,
          ticks: {
            callback: (value) => `${(value * 100).toFixed(0)}%`,
          },
        },
        x: {
          title: {
            display: true,
            text: "Epoch",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`,
          },
        },
      },
    },
  });
}

function renderSampleWindows(sampleWindows) {
  const container = document.querySelector("#sample-table");
  if (!container) return;
  createTable(container, sampleWindows, [
    { key: "sample_index", label: "Sample" },
    { key: "label", label: "True label" },
    { key: "window_bits", label: "Window bits" },
  ]);
}

function renderPredictionSummary(predictions) {
  const container = document.querySelector("#prediction-summary");
  if (!container) return;

  const totals = predictions.reduce(
    (acc, row) => {
      const truth = Number(row.true_label);
      const pred = Number(row.predicted_label);
      if (truth === 1) {
        acc.truePositives += pred === 1 ? 1 : 0;
        acc.falseNegatives += pred === 0 ? 1 : 0;
      } else {
        acc.trueNegatives += pred === 0 ? 1 : 0;
        acc.falsePositives += pred === 1 ? 1 : 0;
      }
      return acc;
    },
    { truePositives: 0, trueNegatives: 0, falsePositives: 0, falseNegatives: 0 }
  );

  const total = predictions.length;
  const accuracy = ((totals.truePositives + totals.trueNegatives) / total) * 100;

  container.innerHTML = `
    <div class="card-grid">
      <div class="card">
        <h3>Overall accuracy</h3>
        <p><strong>${accuracy.toFixed(2)}%</strong> across ${total} samples</p>
      </div>
      <div class="card">
        <h3>Positive windows</h3>
        <p>TP: ${totals.truePositives}<br>FN: ${totals.falseNegatives}</p>
      </div>
      <div class="card">
        <h3>Negative windows</h3>
        <p>TN: ${totals.trueNegatives}<br>FP: ${totals.falsePositives}</p>
      </div>
    </div>
  `;
}

if (document.readyState !== "loading") {
  if (document.body.dataset.page === "maritime") {
    loadMaritimeDataset().catch((error) => {
      console.error(error);
    });
  }
} else {
  document.addEventListener("DOMContentLoaded", () => {
    if (document.body.dataset.page === "maritime") {
      loadMaritimeDataset().catch((error) => {
        console.error(error);
      });
    }
  });
}
