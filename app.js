(function () {
  const D = window.LST_DATA;
  const YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028];
  const min = D.min, max = D.max;
  const points = D.points;
  const n = points.length;

  function toC(v) { return v * (max - min) + min; }
  function toN(c) { return (c - min) / (max - min); }

  const actualC = points.map((p) => p.slice(2, 13));
  const actualN = actualC.map((row) => row.map(toN));

  const $ = (id) => document.getElementById(id);
  const status = (msg) => { $("status").textContent = msg; };

  function colorLST(c) {
    const t = Math.max(0, Math.min(1, (c - 28) / 16));
    const stops = [
      [43, 108, 176],
      [72, 187, 120],
      [236, 201, 75],
      [237, 137, 54],
      [197, 48, 48]
    ];
    const x = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    const r = Math.round(a[0] + (b[0] - a[0]) * f);
    const g = Math.round(a[1] + (b[1] - a[1]) * f);
    const bl = Math.round(a[2] + (b[2] - a[2]) * f);
    return `rgb(${r},${g},${bl})`;
  }

  function evalSplit(predsN, idxs) {
    const yt = [], yp = [];
    for (let i = 0; i < n; i++) {
      for (const t of idxs) {
        yt.push(actualC[i][t]);
        yp.push(toC(predsN[i][t]));
      }
    }
    return LSTModels.metrics(yt, yp);
  }

  let trained = null;
  let map, layerGroup, selected = 0, chart;
  let heatLayer = null;

  function statsActual(yearIdx) {
    const vals = actualC.map((r) => r[Math.min(yearIdx, 10)]);
    return {
      mean: LSTModels.mean(vals),
      min: Math.min.apply(null, vals),
      max: Math.max.apply(null, vals)
    };
  }

  function fmt(x, d) { return Number(x).toFixed(d); }

  function fillKpis() {
    const s = statsActual(6);
    $("kpiPoints").textContent = n.toLocaleString("vi-VN");
    $("kpiYears").textContent = "2016–2026";
    $("kpiMean").textContent = fmt(s.mean, 2) + " °C";
    $("kpiRange").textContent = fmt(min, 1) + "–" + fmt(max, 1);
  }

  function renderTable(tbodyId, splitIdxs) {
    const tb = $(tbodyId);
    tb.innerHTML = "";
    let best = 0, bestRmse = Infinity;
    const rows = trained.ids.map((id, k) => {
      const m = evalSplit(trained.preds[id], splitIdxs);
      if (m.rmse < bestRmse) { bestRmse = m.rmse; best = k; }
      return { id, m };
    });
    rows.forEach((row, k) => {
      const m = row.m;
      const tr = document.createElement("tr");
      if (k === best) tr.className = "best";
      tr.innerHTML = `<td>${trained.names[row.id]}${k === best ? " ★" : ""}</td>
        <td>${fmt(m.mae, 3)}</td><td>${fmt(m.rmse, 3)}</td>
        <td>${fmt(m.r2, 3)}</td><td>${fmt(m.mape, 2)}</td>`;
      tb.appendChild(tr);
    });
    return trained.ids[best];
  }

  function currentYearIdx() {
    return Number($("yearSel").value);
  }
  function currentModel() {
    return $("modelSel").value;
  }
  function currentMode() {
    return $("modeSel").value;
  }

  function valueAt(i, t, mode, modelId) {
    if (t <= 10 && mode === "actual") return actualC[i][t];
    const p = trained.preds[modelId][i][t];
    if (mode === "pred" || t > 10) return toC(p);
    const a = actualC[i][Math.min(t, 10)];
    return a - toC(p);
  }

  function drawMap() {
    if (!layerGroup) return;
    layerGroup.clearLayers();
    const t = currentYearIdx();
    const mode = currentMode();
    const modelId = currentModel();
    const isErr = mode === "error" && t <= 10;
    for (let i = 0; i < n; i++) {
      const v = valueAt(i, t, mode, modelId);
      let col, radius = 7;
      if (isErr) {
        const mag = Math.min(1, Math.abs(v) / 4);
        col = v >= 0 ? `rgb(${Math.round(40 + mag * 180)},70,70)` : `rgb(40,70,${Math.round(40 + mag * 180)})`;
      } else col = colorLST(v);
      const m = L.circleMarker([points[i][1], points[i][0]], {
        radius, color: "#fff", weight: 0.4, fillColor: col, fillOpacity: 0.85
      });
      m.on("click", () => { selected = i; highlightPoint(); drawChart(); });
      layerGroup.addLayer(m);
    }
    highlightPoint();
    const y = YEARS[t];
    $("mapTitle").textContent = mode === "actual"
      ? `LST thực tế ${y} (°C)`
      : mode === "pred"
        ? `LST dự báo ${trained.names[modelId]} — ${y} (°C)`
        : `Sai số (thực tế − dự báo) ${y} (°C)`;
  }

  function highlightPoint() {
    const p = points[selected];
    $("ptInfo").innerHTML = `Điểm #${selected + 1} · lon ${p[0].toFixed(5)}, lat ${p[1].toFixed(5)}`;
  }

  function drawChart() {
    const modelId = currentModel();
    const labels = YEARS.map(String);
    const actual = YEARS.map((_, t) => t <= 10 ? actualC[selected][t] : null);
    const pred = trained.preds[modelId][selected].map(toC);
    if (chart) chart.destroy();
    chart = new Chart($("chart"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "LST thực tế (°C)", data: actual, borderColor: "#12263f", backgroundColor: "transparent", spanGaps: false, tension: 0.15, pointRadius: 3 },
          { label: trained.names[modelId] + " (°C)", data: pred, borderColor: "#1aa79a", borderDash: [6, 4], backgroundColor: "transparent", tension: 0.15, pointRadius: 3 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          title: { display: true, text: "Chuỗi thời gian tại điểm đã chọn · vạch dọc: Train | Val | Test | Dự báo" }
        },
        scales: { y: { title: { display: true, text: "LST (°C)" } } }
      }
    });
  }

  function initMap() {
    map = L.map("map").setView([10.76, 106.66], 10);
    setTimeout(() => map.invalidateSize(), 200);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 16, attribution: "&copy; OpenStreetMap"
    }).addTo(map);
    layerGroup = L.layerGroup().addTo(map);
    const legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML = "<b>LST (°C)</b><div class='bar'></div><div style='display:flex;justify-content:space-between'><span>28</span><span>36</span><span>44</span></div>";
      return div;
    };
    legend.addTo(map);
  }

  function fillModelSelect() {
    const sel = $("modelSel");
    sel.innerHTML = trained.ids.map((id) => `<option value="${id}">${trained.names[id]}</option>`).join("");
  }

  function fillYearSelect() {
    $("yearSel").innerHTML = YEARS.map((y, i) => `<option value="${i}" ${y === 2025 ? "selected" : ""}>${y}${y >= 2027 ? " (dự báo)" : ""}</option>`).join("");
  }

  $("yearSel").addEventListener("change", drawMap);
  $("modelSel").addEventListener("change", () => { drawMap(); drawChart(); });
  $("modeSel").addEventListener("change", drawMap);

  fillKpis();
  initMap();
  fillYearSelect();
  status("Đang huấn luyện 6 mô hình trên tập Train (2016–2022) trong trình duyệt…");

  setTimeout(() => {
    const t0 = performance.now();
    trained = LSTModels.train(points, actualN, status);
    const ms = Math.round(performance.now() - t0);
    fillModelSelect();
    const bestTest = renderTable("tblTest", LSTModels.TEST);
    renderTable("tblVal", LSTModels.VAL);
    $("modelSel").value = bestTest;
    status(`Huấn luyện xong trong ${(ms / 1000).toFixed(1)} giây · mô hình tốt nhất trên Test (MAE/RMSE): ${trained.names[bestTest]}. Kết quả đã khôi phục về °C bằng scaler min=${min.toFixed(2)}, max=${max.toFixed(2)}.`);
    drawMap();
    const b = L.latLngBounds(points.map((p) => [p[1], p[0]]));
    map.invalidateSize();
    map.fitBounds(b.pad(0.06));
    drawChart();
  }, 40);
})();
