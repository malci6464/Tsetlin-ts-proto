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
  return rows
    .filter(Boolean)
    .map((row) => {
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
  entries
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .forEach(([label, value]) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${label}:</strong> ${value}`;
      fragment.appendChild(li);
    });
  target.appendChild(fragment);
}

function createTable(target, rows, columns) {
  target.innerHTML = "";
  if (!rows.length) {
    target.textContent = "No samples available.";
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(({ label }) => {
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

function countActiveBits(bitString) {
  if (typeof bitString !== "string" || bitString.length === 0) {
    return 0;
  }
  return [...bitString].reduce((acc, bit) => acc + (bit === "1" ? 1 : 0), 0);
}

const SCENARIO_CONFIG = {
  maritime: {
    basePath: "time_series_outputs/maritime/20251003_125026",
    sampleTableTarget: "#sample-table",
    sampleLimit: 20,
    sampleColumns: [
      { key: "sample_index", label: "Sample" },
      { key: "label", label: "True label" },
      { key: "window_bits", label: "Window bits" },
    ],
    predictionType: "classification",
    metadataEntries: (summary) => {
      const metadata = summary.metadata || {};
      return [
        ["Scenario", summary.scenario ?? "maritime"],
        ["Sensor count", metadata.sensors?.length ?? 0],
        ["Window size", metadata.window_size],
        ["Noise probability", metadata.noise_probability],
        ["Event probability", metadata.event_probability],
        ["Sequences", metadata.n_sequences],
        ["Sequence length", metadata.sequence_length],
        ["Total samples", metadata.samples],
        ["Positive windows", metadata.positives],
      ];
    },
    accuracyChartOptions: {
      scales: {
        y: {
          min: 0.85,
          max: 1,
        },
      },
    },
  },
  temperature: {
    basePath: "time_series_outputs/temperature/20251005_090000",
    sampleTableTarget: "#temperature-sample-table",
    sampleColumns: [
      { key: "timestamp", label: "Timestamp" },
      { key: "season", label: "Season" },
      {
        key: "observed_temp",
        label: "Observed (°C)",
        render: (value) => Number(value).toFixed(1),
      },
      {
        key: "predicted_temp",
        label: "Predicted (°C)",
        render: (value) => Number(value).toFixed(1),
      },
      { key: "window_bits", label: "Sensor fingerprint" },
    ],
    sampleLimit: 15,
    predictionType: "regression",
    predictionSummaryTarget: "#prediction-summary",
    predictionChartTarget: "#temperature-prediction-chart",
    metadataEntries: (summary) => {
      const metadata = summary.metadata || {};
      return [
        ["Scenario", "Seasonal temperature forecasting"],
        ["Location", metadata.location],
        ["Window size (hours)", metadata.window_size],
        ["Sequences", metadata.n_sequences],
        ["Samples", metadata.samples],
        ["Heat alerts (>28°C)", metadata.positives],
        ["Target", metadata.target],
        ["Noise probability", metadata.noise_probability],
      ];
    },
    accuracyChartOptions: {
      scales: {
        y: {
          min: 0.7,
          max: 0.95,
        },
      },
    },
    extraCsv: {
      temperatureSeries: "temperature_series.csv",
      dailyProfile: "daily_profile.csv",
    },
    onExtraDataLoaded: (extraData) => {
      if (extraData.temperatureSeries) {
        renderTemperatureSeries(extraData.temperatureSeries);
      }
      if (extraData.dailyProfile) {
        renderDailyProfile(extraData.dailyProfile);
      }
    },
    onAfterRender: () => {
      renderModelComparisonChart();
    },
  },
  industrial: {
    basePath: "time_series_outputs/industrial/20251004_155824",
    sampleTableTarget: "#industrial-sample-table",
    sampleLimit: 10,
    sampleColumns: [
      { key: "sample_index", label: "Snapshot" },
      {
        key: "label",
        label: "Alarm",
        render: (value) => (Number(value) === 1 ? "Triggered" : "Normal"),
      },
      {
        key: "window_bits",
        label: "Active sensors",
        render: (value) => countActiveBits(value),
      },
    ],
    predictionType: "classification",
    predictionSummaryTarget: "#industrial-prediction-summary",
    metadataTarget: "#industrial-metadata",
    hyperparameterTarget: "#industrial-hyperparameters",
    sensorListTarget: "#industrial-hazard-rules",
    sensorListEntries: (summary) => {
      const metadata = summary.metadata || {};
      const ruleCounts = metadata.rule_counts || {};
      const rules = metadata.hazard_rules || [];
      return rules.map((rule) => {
        const count = ruleCounts[rule] ?? 0;
        return `<strong>${rule}</strong><span>${count} alarms</span>`;
      });
    },
    metadataEntries: (summary) => {
      const metadata = summary.metadata || {};
      return [
        ["Scenario", "Industrial anomaly detection"],
        ["Sensor channels", metadata.sensor_count],
        ["Samples", metadata.samples],
        ["Snapshot interval", `${metadata.snapshot_interval_seconds ?? "?"} s`],
        ["Hazard probability", metadata.hazard_probability],
        ["Noise probability", metadata.noise_probability],
      ];
    },
    scalingCsv: "sensor_scaling.csv",
    scalingCharts: {
      throughput: "#scaling-throughput-chart",
      memory: "#scaling-memory-chart",
    },
    scalingTableTarget: "#scaling-table",
    onScalingDataLoaded: (rows, context) => {
      renderSensorScalingCharts(rows, context.config);
      renderSensorScalingTable(rows, context.config);
    },
  },
};

async function initializePage() {
  const page = document.body?.dataset?.page;
  if (!page || !SCENARIO_CONFIG[page]) {
    return;
  }

  const config = SCENARIO_CONFIG[page];
  const basePath = config.basePath;

  const [summary, accuracyHistory, sampleWindows, predictions] = await Promise.all([
    fetchJSON(`${basePath}/run_summary.json`),
    fetchCSV(`${basePath}/accuracy_history.csv`),
    fetchCSV(`${basePath}/sample_windows.csv`),
    fetchCSV(`${basePath}/test_predictions.csv`),
  ]);

  populateSummary(summary, config);
  renderAccuracyChart(accuracyHistory, config);
  const limit = config.sampleLimit ?? sampleWindows.length;
  renderSampleWindows(sampleWindows.slice(0, limit), config);
  renderPredictionSummary(predictions, config);

  let extraData = {};
  if (config.extraCsv) {
    const entries = Object.entries(config.extraCsv);
    const results = await Promise.all(
      entries.map(async ([key, file]) => {
        const data = await fetchCSV(`${basePath}/${file}`);
        return [key, data];
      })
    );
    extraData = Object.fromEntries(results);
    if (typeof config.onExtraDataLoaded === "function") {
      config.onExtraDataLoaded(extraData, { summary, accuracyHistory, sampleWindows, predictions, config });
    }
  }

  if (config.scalingCsv) {
    const scalingRows = await fetchCSV(`${basePath}/${config.scalingCsv}`);
    if (typeof config.onScalingDataLoaded === "function") {
      config.onScalingDataLoaded(scalingRows, { summary, config });
    } else {
      renderSensorScalingCharts(scalingRows, config);
      renderSensorScalingTable(scalingRows, config);
    }
  }

  if (typeof config.onAfterRender === "function") {
    config.onAfterRender({ summary, accuracyHistory, sampleWindows, predictions, extraData, config });
  }
}

function populateSummary(summary, config = {}) {
  const metadata = summary.metadata || {};
  const metadataList = document.querySelector(config.metadataTarget ?? "#metadata-list");
  if (metadataList) {
    const entries = typeof config.metadataEntries === "function"
      ? config.metadataEntries(summary)
      : [
          ["Scenario", summary.scenario ?? "unknown"],
          ["Sensor count", metadata.sensors?.length ?? 0],
          ["Window size", metadata.window_size],
          ["Noise probability", metadata.noise_probability],
          ["Event probability", metadata.event_probability],
          ["Sequences", metadata.n_sequences],
          ["Sequence length", metadata.sequence_length],
          ["Total samples", metadata.samples],
          ["Positive windows", metadata.positives],
        ];
    createSummaryList(metadataList, entries);
  }

  const sensorList = document.querySelector(config.sensorListTarget ?? "#sensor-list");
  if (sensorList) {
    if (typeof config.sensorListEntries === "function") {
      const entries = config.sensorListEntries(summary);
      sensorList.innerHTML = entries
        .map((entry) => `<li>${entry}</li>`)
        .join("");
    } else if (Array.isArray(metadata.sensors)) {
      sensorList.innerHTML = metadata.sensors.map((sensor) => `<li>${sensor}</li>`).join("");
    }
  }

  const hyperparameterList = document.querySelector(config.hyperparameterTarget ?? "#hyperparameter-list");
  if (hyperparameterList) {
    const entries = typeof config.hyperparameterEntries === "function"
      ? config.hyperparameterEntries(summary)
      : Object.entries(summary.hyperparameters || {});
    createSummaryList(hyperparameterList, entries);
  }

  const accuracyBadges = document.querySelector(config.accuracyBadgesTarget ?? "#accuracy-badges");
  if (accuracyBadges) {
    const { training } = summary;
    const finalAcc = training?.final_test_accuracy;
    const bestAcc = training?.best_test_accuracy;
    if (typeof finalAcc === "number" && typeof bestAcc === "number") {
      accuracyBadges.innerHTML = `
        <span class="badge">Final test accuracy: ${(finalAcc * 100).toFixed(2)}%</span>
        <span class="badge">Best test accuracy: ${(bestAcc * 100).toFixed(2)}%</span>
      `;
    }
  }
}

function renderAccuracyChart(accuracyHistory, config = {}) {
  const ctx = document.querySelector(config.accuracyChartTarget ?? "#accuracy-chart");
  if (!ctx) return;

  const epochs = accuracyHistory.map((row) => Number(row.epoch));
  const trainAcc = accuracyHistory.map((row) => Number(row.train_accuracy));
  const testAcc = accuracyHistory.map((row) => Number(row.test_accuracy));

  const options = {
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
          min: 0,
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
  };

  if (config.accuracyChartOptions) {
    const { scales, ...rest } = config.accuracyChartOptions;
    Object.assign(options.options, rest);
    if (scales?.y) {
      options.options.scales.y = { ...options.options.scales.y, ...scales.y };
    }
    if (scales?.x) {
      options.options.scales.x = { ...options.options.scales.x, ...scales.x };
    }
  }

  new Chart(ctx, options);
}

function renderSampleWindows(sampleWindows, config = {}) {
  const targetSelector = config.sampleTableTarget ?? "#sample-table";
  const container = document.querySelector(targetSelector);
  if (!container) return;
  const columns = config.sampleColumns ?? [
    { key: "sample_index", label: "Sample" },
    { key: "label", label: "True label" },
    { key: "window_bits", label: "Window bits" },
  ];
  createTable(container, sampleWindows, columns);
}

function renderPredictionSummary(predictions, config = {}) {
  const targetSelector = config.predictionSummaryTarget ?? "#prediction-summary";
  const container = document.querySelector(targetSelector);
  if (!container) return;

  if ((config.predictionType ?? "classification") === "regression") {
    if (!predictions.length) {
      container.textContent = "No predictions available.";
      return;
    }

    const trueValues = predictions.map((row) => Number(row.true_temp));
    const predictedValues = predictions.map((row) => Number(row.predicted_temp));
    const errors = trueValues.map((truth, idx) => truth - predictedValues[idx]);

    const total = errors.length;
    const mae = errors.reduce((acc, value) => acc + Math.abs(value), 0) / total;
    const rmse = Math.sqrt(errors.reduce((acc, value) => acc + value * value, 0) / total);
    const bias = errors.reduce((acc, value) => acc + value, 0) / total;
    const withinOne = (errors.filter((value) => Math.abs(value) <= 1).length / total) * 100;

    container.innerHTML = `
      <div class="card-grid">
        <div class="card">
          <h3>MAE</h3>
          <p><strong>${mae.toFixed(2)} °C</strong></p>
        </div>
        <div class="card">
          <h3>RMSE</h3>
          <p><strong>${rmse.toFixed(2)} °C</strong></p>
        </div>
        <div class="card">
          <h3>Bias</h3>
          <p><strong>${bias.toFixed(2)} °C</strong></p>
        </div>
        <div class="card">
          <h3>Within ±1 °C</h3>
          <p><strong>${withinOne.toFixed(1)}%</strong> of predictions</p>
        </div>
      </div>
    `;

    if (config.predictionChartTarget) {
      renderRegressionPredictionChart(predictions, config.predictionChartTarget);
    }
    return;
  }

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

function renderSensorScalingCharts(rows, config = {}) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return;
  }

  const sorted = [...rows].sort((a, b) => Number(a.sensor_count) - Number(b.sensor_count));
  const sensorCounts = sorted.map((row) => Number(row.sensor_count));
  const throughput = sorted.map((row) => Number(row.samples_per_second));
  const latencyMs = sorted.map((row) => Number(row.per_sample_seconds) * 1000);
  const rssBaselineMiB = sorted.map((row) => Number(row.rss_baseline_bytes) / 1_048_576);
  const rssPeakMiB = sorted.map((row) => Number(row.rss_peak_bytes) / 1_048_576);
  const rssFinalMiB = sorted.map((row) => Number(row.rss_final_bytes) / 1_048_576);

  const throughputTarget = config.scalingCharts?.throughput ?? "#scaling-throughput-chart";
  const memoryTarget = config.scalingCharts?.memory ?? "#scaling-memory-chart";

  const throughputCanvas = document.querySelector(throughputTarget);
  if (throughputCanvas) {
    new Chart(throughputCanvas, {
      type: "line",
      data: {
        labels: sensorCounts,
        datasets: [
          {
            label: "Throughput (snapshots/s)",
            data: throughput,
            borderColor: "#0ea5e9",
            backgroundColor: "rgba(14, 165, 233, 0.2)",
            tension: 0.25,
            fill: true,
            yAxisID: "y",
          },
          {
            label: "Latency (ms)",
            data: latencyMs,
            borderColor: "#f97316",
            backgroundColor: "rgba(249, 115, 22, 0.2)",
            tension: 0.25,
            fill: true,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        stacked: false,
        scales: {
          x: {
            title: { display: true, text: "Active sensor channels" },
          },
          y: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Snapshots per second" },
          },
          y1: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Latency per snapshot (ms)" },
            grid: { drawOnChartArea: false },
          },
        },
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });
  }

  const memoryCanvas = document.querySelector(memoryTarget);
  if (memoryCanvas) {
    new Chart(memoryCanvas, {
      type: "line",
      data: {
        labels: sensorCounts,
        datasets: [
          {
            label: "Baseline RSS (MiB)",
            data: rssBaselineMiB,
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139, 92, 246, 0.2)",
            tension: 0.25,
            fill: true,
          },
          {
            label: "Peak RSS (MiB)",
            data: rssPeakMiB,
            borderColor: "#ef4444",
            backgroundColor: "rgba(239, 68, 68, 0.2)",
            tension: 0.25,
            fill: true,
          },
          {
            label: "Post-benchmark RSS (MiB)",
            data: rssFinalMiB,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.15)",
            tension: 0.25,
            fill: true,
            borderDash: [6, 4],
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        stacked: false,
        scales: {
          x: {
            title: { display: true, text: "Active sensor channels" },
          },
          y: {
            title: { display: true, text: "Resident set size (MiB)" },
          },
        },
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });
  }
}

function renderSensorScalingTable(rows, config = {}) {
  const targetSelector = config.scalingTableTarget ?? "#scaling-table";
  const container = document.querySelector(targetSelector);
  if (!container || !Array.isArray(rows) || rows.length === 0) {
    return;
  }

  const processed = rows
    .map((row) => ({
      sensor_count: Number(row.sensor_count),
      samples_per_second: Number(row.samples_per_second),
      per_sample_ms: Number(row.per_sample_seconds) * 1000,
      rss_baseline_mib: Number(row.rss_baseline_bytes) / 1_048_576,
      rss_peak_mib: Number(row.rss_peak_bytes) / 1_048_576,
      rss_final_mib: Number(row.rss_final_bytes) / 1_048_576,
      window_utilisation: Number(row.window_utilisation) * 100,
      train_time_seconds: Number(row.train_time_seconds),
      test_accuracy: Number(row.test_accuracy) * 100,
    }))
    .sort((a, b) => a.sensor_count - b.sensor_count);

  const columns = [
    { key: "sensor_count", label: "Sensors" },
    {
      key: "samples_per_second",
      label: "Throughput (snapshots/s)",
      render: (value) => value.toFixed(2),
    },
    {
      key: "per_sample_ms",
      label: "Latency (ms)",
      render: (value) => value.toFixed(3),
    },
    {
      key: "rss_baseline_mib",
      label: "Baseline RSS (MiB)",
      render: (value) => value.toFixed(2),
    },
    {
      key: "rss_peak_mib",
      label: "Peak RSS (MiB)",
      render: (value) => value.toFixed(2),
    },
    {
      key: "rss_final_mib",
      label: "Post-benchmark RSS (MiB)",
      render: (value) => value.toFixed(2),
    },
    {
      key: "window_utilisation",
      label: "Window utilisation (%)",
      render: (value) => value.toFixed(2),
    },
    {
      key: "train_time_seconds",
      label: "Training time (s)",
      render: (value) => value.toFixed(1),
    },
    {
      key: "test_accuracy",
      label: "Test accuracy (%)",
      render: (value) => value.toFixed(2),
    },
  ];

  createTable(container, processed, columns);
}

function renderRegressionPredictionChart(predictions, targetSelector) {
  const ctx = document.querySelector(targetSelector);
  if (!ctx) return;

  const subset = predictions.slice(0, 120);
  const labels = subset.map((row) => {
    if (!row.timestamp) {
      return `Sample ${row.sample_index ?? ""}`;
    }
    const [date, time = ""] = row.timestamp.split("T");
    return `${date} ${time.slice(0, 5)}`.trim();
  });
  const truth = subset.map((row) => Number(row.true_temp));
  const predicted = subset.map((row) => Number(row.predicted_temp));

  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Observed",
          data: truth,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.12)",
          tension: 0.2,
          fill: true,
        },
        {
          label: "Predicted",
          data: predicted,
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.12)",
          tension: 0.2,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: "Temperature (°C)",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)} °C`,
          },
        },
      },
    },
  });
}

function renderTemperatureSeries(series) {
  const ctx = document.querySelector("#temperature-series-chart");
  if (!ctx) return;

  const subset = series.slice(0, 360);
  const labels = subset.map((row) => {
    if (!row.timestamp) {
      return "";
    }
    return row.timestamp.replace("T", " ").slice(0, 16);
  });
  const observed = subset.map((row) => Number(row.observed_temp));
  const predicted = subset.map((row) => Number(row.predicted_temp));
  const seasonal = subset.map((row) => Number(row.seasonal_component));
  const daily = subset.map((row) => Number(row.daily_component));

  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Observed temperature",
          data: observed,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          tension: 0.25,
          fill: false,
        },
        {
          label: "Predicted temperature",
          data: predicted,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.1)",
          tension: 0.25,
          fill: false,
        },
        {
          label: "Seasonal component",
          data: seasonal,
          borderDash: [6, 4],
          borderColor: "#f59e0b",
          tension: 0.2,
          fill: false,
        },
        {
          label: "Daily component",
          data: daily,
          borderDash: [4, 4],
          borderColor: "#0ea5e9",
          tension: 0.2,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 12,
          },
        },
        y: {
          title: {
            display: true,
            text: "Temperature (°C)",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function renderDailyProfile(profile) {
  const ctx = document.querySelector("#daily-profile-chart");
  if (!ctx) return;

  const labels = profile.map((row) => `${row.hour}:00`);
  const observed = profile.map((row) => Number(row.observed_temp));
  const predicted = profile.map((row) => Number(row.predicted_temp));

  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Observed",
          data: observed,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.15)",
          tension: 0.25,
          fill: true,
        },
        {
          label: "Predicted",
          data: predicted,
          borderColor: "#a855f7",
          backgroundColor: "rgba(168, 85, 247, 0.12)",
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          title: {
            display: true,
            text: "Temperature (°C)",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function renderModelComparisonChart() {
  const ctx = document.querySelector("#model-comparison-chart");
  if (!ctx) return;

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Tsetlin", "GRU", "Transformer"],
      datasets: [
        {
          label: "Parameters (thousands)",
          data: [12, 480, 1200],
          backgroundColor: "rgba(59, 130, 246, 0.65)",
        },
        {
          label: "Training energy (relative)",
          data: [1, 14, 22],
          backgroundColor: "rgba(244, 114, 182, 0.65)",
        },
        {
          label: "Accuracy (%)",
          data: [92, 94, 95],
          backgroundColor: "rgba(16, 185, 129, 0.65)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const label = context.dataset.label;
              const value = context.parsed.y;
              if (label.includes("Accuracy")) {
                return `${label}: ${value.toFixed(1)}%`;
              }
              if (label.includes("Parameters")) {
                return `${label}: ${value.toFixed(0)}k`;
              }
              return `${label}: ${value.toFixed(1)}×`;
            },
          },
        },
      },
    },
  });
}

if (document.readyState !== "loading") {
  initializePage().catch((error) => console.error(error));
} else {
  document.addEventListener("DOMContentLoaded", () => {
    initializePage().catch((error) => console.error(error));
  });
}
