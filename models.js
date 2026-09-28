/* Mô hình AI gọn chạy trên trình duyệt — cùng split Train 2016–2022 / Val 2023–2024 / Test 2025–2026 */
(function (global) {
  const TRAIN = [0, 1, 2, 3, 4, 5, 6];
  const VAL = [7, 8];
  const TEST = [9, 10];
  const LOOK = 4;

  function rng(seed) {
    let s = seed | 0;
    return function () {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  function mean(a) {
    let t = 0;
    for (let i = 0; i < a.length; i++) t += a[i];
    return t / a.length;
  }

  function metrics(yTrue, yPred) {
    const n = yTrue.length;
    let mae = 0, se = 0, mape = 0, yt = 0, yt2 = 0;
    for (let i = 0; i < n; i++) {
      const e = yPred[i] - yTrue[i];
      mae += Math.abs(e);
      se += e * e;
      mape += Math.abs(e / Math.max(1e-6, Math.abs(yTrue[i])));
      yt += yTrue[i];
      yt2 += yTrue[i] * yTrue[i];
    }
    mae /= n;
    const rmse = Math.sqrt(se / n);
    const ybar = yt / n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += (yTrue[i] - ybar) * (yTrue[i] - ybar);
    const r2 = 1 - se / Math.max(1e-12, ssTot);
    return { mae, rmse, r2, mape: (mape / n) * 100 };
  }

  function fitLine(xs, ys) {
    const n = xs.length;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
    }
    const den = n * sxx - sx * sx;
    const b = den === 0 ? 0 : (n * sxy - sx * sy) / den;
    const a = (sy - b * sx) / n;
    return { a, b };
  }

  function fitQuad(xs, ys) {
    const n = xs.length;
    let s1 = n, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
    for (let i = 0; i < n; i++) {
      const x = xs[i], y = ys[i], x2 = x * x;
      sx += x; sx2 += x2; sx3 += x2 * x; sx4 += x2 * x2;
      sy += y; sxy += x * y; sx2y += x2 * y;
    }
    function solve3(m, v) {
      const a = m.map((row) => row.slice());
      const b = v.slice();
      for (let i = 0; i < 3; i++) {
        let p = i;
        for (let r = i + 1; r < 3; r++) if (Math.abs(a[r][i]) > Math.abs(a[p][i])) p = r;
        [a[i], a[p]] = [a[p], a[i]];
        [b[i], b[p]] = [b[p], b[i]];
        const piv = a[i][i] || 1e-12;
        for (let j = i; j < 3; j++) a[i][j] /= piv;
        b[i] /= piv;
        for (let r = 0; r < 3; r++) if (r !== i) {
          const f = a[r][i];
          for (let j = i; j < 3; j++) a[r][j] -= f * a[i][j];
          b[r] -= f * b[i];
        }
      }
      return b;
    }
    return solve3(
      [[s1, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]],
      [sy, sxy, sx2y]
    );
  }

  function extraTrees(X, y, opts) {
    const nTree = opts.nTree || 24;
    const depth = opts.depth || 5;
    const minLeaf = opts.minLeaf || 18;
    const rand = rng(opts.seed || 7);
    const n = y.length;
    const d = X[0].length;

    function leafVal(idx) {
      let s = 0;
      for (let i = 0; i < idx.length; i++) s += y[idx[i]];
      return s / idx.length;
    }

    function build(idx, dep) {
      if (dep <= 0 || idx.length <= minLeaf) return { v: leafVal(idx) };
      const feat = Math.floor(rand() * d);
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < idx.length; i++) {
        const v = X[idx[i]][feat];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (mx - mn < 1e-12) return { v: leafVal(idx) };
      const thr = mn + (mx - mn) * (0.2 + 0.6 * rand());
      const L = [], R = [];
      for (let i = 0; i < idx.length; i++) {
        if (X[idx[i]][feat] <= thr) L.push(idx[i]);
        else R.push(idx[i]);
      }
      if (!L.length || !R.length) return { v: leafVal(idx) };
      return { f: feat, t: thr, L: build(L, dep - 1), R: build(R, dep - 1) };
    }

    function predictOne(tree, x) {
      let node = tree;
      while (node.L) node = x[node.f] <= node.t ? node.L : node.R;
      return node.v;
    }

    const idx0 = Array.from({ length: n }, (_, i) => i);
    const trees = [];
    for (let t = 0; t < nTree; t++) {
      const bag = [];
      for (let i = 0; i < n; i++) bag.push(idx0[Math.floor(rand() * n)]);
      trees.push(build(bag, depth));
    }
    return {
      predict(x) {
        let s = 0;
        for (let i = 0; i < trees.length; i++) s += predictOne(trees[i], x);
        return s / trees.length;
      }
    };
  }

  function gboost(X, y, opts) {
    const M = opts.rounds || 40;
    const lr = opts.lr || 0.12;
    const rand = rng(opts.seed || 11);
    const n = y.length;
    const d = X[0].length;
    const pred = new Float64Array(n);
    const yMean = mean(y);
    for (let i = 0; i < n; i++) pred[i] = yMean;
    const stumps = [];

    function stump(res) {
      let best = { feat: 0, thr: 0, left: 0, right: 0, loss: Infinity };
      for (let k = 0; k < 8; k++) {
        const feat = Math.floor(rand() * d);
        const i1 = Math.floor(rand() * n), i2 = Math.floor(rand() * n);
        const thr = (X[i1][feat] + X[i2][feat]) / 2;
        let sl = 0, nl = 0, sr = 0, nr = 0;
        for (let i = 0; i < n; i++) {
          if (X[i][feat] <= thr) { sl += res[i]; nl++; }
          else { sr += res[i]; nr++; }
        }
        if (!nl || !nr) continue;
        const left = sl / nl, right = sr / nr;
        let loss = 0;
        for (let i = 0; i < n; i++) {
          const p = X[i][feat] <= thr ? left : right;
          const e = res[i] - p;
          loss += e * e;
        }
        if (loss < best.loss) best = { feat, thr, left, right, loss };
      }
      return best;
    }

    const res = new Float64Array(n);
    for (let m = 0; m < M; m++) {
      for (let i = 0; i < n; i++) res[i] = y[i] - pred[i];
      const s = stump(res);
      stumps.push(s);
      for (let i = 0; i < n; i++) pred[i] += lr * (X[i][s.feat] <= s.thr ? s.left : s.right);
    }
    return {
      predict(x) {
        let p = yMean;
        for (let i = 0; i < stumps.length; i++) {
          const s = stumps[i];
          p += lr * (x[s.feat] <= s.thr ? s.left : s.right);
        }
        return p;
      }
    };
  }

  function tanh(x) { return Math.tanh(Math.max(-20, Math.min(20, x))); }
  function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

  function trainElman(seqs, targets, hid, epochs, lr, seed) {
    const rand = rng(seed);
    const Wxh = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.2);
    const Whh = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.2);
    const bh = Array.from({ length: hid }, () => 0);
    const Why = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.2);
    let by = 0;
    const N = seqs.length;
    for (let ep = 0; ep < epochs; ep++) {
      for (let n = 0; n < N; n++) {
        const xs = seqs[n];
        const hHist = [new Float64Array(hid)];
        const aHist = [];
        for (let t = 0; t < xs.length; t++) {
          const hPrev = hHist[t];
          const a = new Float64Array(hid);
          const h = new Float64Array(hid);
          for (let j = 0; j < hid; j++) {
            a[j] = Wxh[j] * xs[t] + Whh[j] * hPrev[j] + bh[j];
            h[j] = tanh(a[j]);
          }
          aHist.push(a);
          hHist.push(h);
        }
        const hT = hHist[xs.length];
        let yhat = by;
        for (let j = 0; j < hid; j++) yhat += Why[j] * hT[j];
        const err = yhat - targets[n];
        const dWhy = new Float64Array(hid);
        let dh = new Float64Array(hid);
        for (let j = 0; j < hid; j++) {
          dWhy[j] = err * hT[j];
          dh[j] = err * Why[j];
        }
        let dby = err;
        const dWxh = new Float64Array(hid);
        const dWhh = new Float64Array(hid);
        const dbh = new Float64Array(hid);
        for (let t = xs.length - 1; t >= 0; t--) {
          const hPrev = hHist[t];
          const a = aHist[t];
          const dhNext = new Float64Array(hid);
          for (let j = 0; j < hid; j++) {
            const dt = dh[j] * (1 - tanh(a[j]) * tanh(a[j]));
            dWxh[j] += dt * xs[t];
            dWhh[j] += dt * hPrev[j];
            dbh[j] += dt;
            dhNext[j] = dt * Whh[j];
          }
          dh = dhNext;
        }
        for (let j = 0; j < hid; j++) {
          Wxh[j] -= lr * dWxh[j];
          Whh[j] -= lr * dWhh[j];
          bh[j] -= lr * dbh[j];
          Why[j] -= lr * dWhy[j];
        }
        by -= lr * dby;
      }
    }
    return {
      predictSeq(xs) {
        let h = new Float64Array(hid);
        for (let t = 0; t < xs.length; t++) {
          const hn = new Float64Array(hid);
          for (let j = 0; j < hid; j++) hn[j] = tanh(Wxh[j] * xs[t] + Whh[j] * h[j] + bh[j]);
          h = hn;
        }
        let yhat = by;
        for (let j = 0; j < hid; j++) yhat += Why[j] * h[j];
        return yhat;
      }
    };
  }

  function trainGRU(seqs, targets, hid, epochs, lr, seed) {
    const rand = rng(seed);
    const Wz = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Uz = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Wr = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Ur = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Wh = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Uh = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    const Why = Array.from({ length: hid }, () => (rand() * 2 - 1) * 0.15);
    let by = 0;
    const N = seqs.length;
    for (let ep = 0; ep < epochs; ep++) {
      for (let n = 0; n < N; n++) {
        const xs = seqs[n];
        let h = new Float64Array(hid);
        const hs = [h];
        for (let t = 0; t < xs.length; t++) {
          const hn = new Float64Array(hid);
          for (let j = 0; j < hid; j++) {
            const z = sigmoid(Wz[j] * xs[t] + Uz[j] * h[j]);
            const r = sigmoid(Wr[j] * xs[t] + Ur[j] * h[j]);
            const ht = tanh(Wh[j] * xs[t] + Uh[j] * r * h[j]);
            hn[j] = (1 - z) * h[j] + z * ht;
          }
          h = hn;
          hs.push(h);
        }
        let yhat = by;
        for (let j = 0; j < hid; j++) yhat += Why[j] * h[j];
        const err = yhat - targets[n];
        for (let j = 0; j < hid; j++) Why[j] -= lr * err * h[j];
        by -= lr * err;
        const dh = new Float64Array(hid);
        for (let j = 0; j < hid; j++) dh[j] = err * Why[j];
        for (let t = xs.length - 1; t >= 0; t--) {
          const x = xs[t];
          const hPrev = hs[t];
          const hCur = hs[t + 1];
          for (let j = 0; j < hid; j++) {
            const z = sigmoid(Wz[j] * x + Uz[j] * hPrev[j]);
            const r = sigmoid(Wr[j] * x + Ur[j] * hPrev[j]);
            const ht = tanh(Wh[j] * x + Uh[j] * r * hPrev[j]);
            const dz = dh[j] * (ht - hPrev[j]) * z * (1 - z);
            const dht = dh[j] * z * (1 - ht * ht);
            const dr = dht * Uh[j] * hPrev[j] * r * (1 - r);
            Wz[j] -= lr * dz * x;
            Uz[j] -= lr * dz * hPrev[j];
            Wr[j] -= lr * dr * x;
            Ur[j] -= lr * dr * hPrev[j];
            Wh[j] -= lr * dht * x;
            Uh[j] -= lr * dht * r * hPrev[j];
            dh[j] = dh[j] * (1 - z) + dz * Uz[j] + dr * Ur[j] + dht * Uh[j] * r;
          }
        }
      }
    }
    return {
      predictSeq(xs) {
        let h = new Float64Array(hid);
        for (let t = 0; t < xs.length; t++) {
          const hn = new Float64Array(hid);
          for (let j = 0; j < hid; j++) {
            const z = sigmoid(Wz[j] * xs[t] + Uz[j] * h[j]);
            const r = sigmoid(Wr[j] * xs[t] + Ur[j] * h[j]);
            const ht = tanh(Wh[j] * xs[t] + Uh[j] * r * h[j]);
            hn[j] = (1 - z) * h[j] + z * ht;
          }
          h = hn;
        }
        let yhat = by;
        for (let j = 0; j < hid; j++) yhat += Why[j] * h[j];
        return yhat;
      }
    };
  }

  function trainAttn(seqs, targets, epochs, lr, seed) {
    const rand = rng(seed);
    const w = Array.from({ length: LOOK }, () => rand() * 0.4 - 0.2);
    let b = 0;
    const N = seqs.length;
    for (let ep = 0; ep < epochs; ep++) {
      for (let n = 0; n < N; n++) {
        const xs = seqs[n];
        let m = -Infinity;
        for (let t = 0; t < LOOK; t++) m = Math.max(m, w[t] * xs[t]);
        const e = new Float64Array(LOOK);
        let s = 0;
        for (let t = 0; t < LOOK; t++) { e[t] = Math.exp(w[t] * xs[t] - m); s += e[t]; }
        let ctx = 0;
        const a = new Float64Array(LOOK);
        for (let t = 0; t < LOOK; t++) { a[t] = e[t] / s; ctx += a[t] * xs[t]; }
        const yhat = ctx + b;
        const err = yhat - targets[n];
        b -= lr * err;
        for (let t = 0; t < LOOK; t++) {
          let dctx = 0;
          for (let k = 0; k < LOOK; k++) {
            const g = a[k] * ((k === t ? 1 : 0) - a[t]);
            dctx += g * xs[k];
          }
          w[t] -= lr * err * (dctx * xs[t] + a[t]);
        }
      }
    }
    return {
      predictSeq(xs) {
        let m = -Infinity;
        for (let t = 0; t < LOOK; t++) m = Math.max(m, w[t] * xs[t]);
        let s = 0, ctx = 0;
        const e = [];
        for (let t = 0; t < LOOK; t++) { e[t] = Math.exp(w[t] * xs[t] - m); s += e[t]; }
        for (let t = 0; t < LOOK; t++) ctx += (e[t] / s) * xs[t];
        return ctx + b;
      }
    };
  }

  function collectWindows(normSeries) {
    const seqs = [], targets = [];
    for (let i = 0; i < normSeries.length; i++) {
      const s = normSeries[i];
      for (let t = LOOK; t <= 6; t++) {
        seqs.push(s.slice(t - LOOK, t));
        targets.push(s[t]);
      }
    }
    return { seqs, targets };
  }

  function collectTabular(points, normSeries) {
    const X = [], y = [];
    let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
    for (let i = 0; i < points.length; i++) {
      lonMin = Math.min(lonMin, points[i][0]);
      lonMax = Math.max(lonMax, points[i][0]);
      latMin = Math.min(latMin, points[i][1]);
      latMax = Math.max(latMax, points[i][1]);
    }
    for (let i = 0; i < points.length; i++) {
      const s = normSeries[i];
      const lon = (points[i][0] - lonMin) / (lonMax - lonMin);
      const lat = (points[i][1] - latMin) / (latMax - latMin);
      for (let t = 2; t <= 6; t++) {
        X.push([lon, lat, t / 10, s[t - 1], s[t - 2]]);
        y.push(s[t]);
      }
    }
    return { X, y, lonMin, lonMax, latMin, latMax };
  }

  function stepFeat(point, s, t, box) {
    const lon = (point[0] - box.lonMin) / (box.lonMax - box.lonMin);
    const lat = (point[1] - box.latMin) / (box.latMax - box.latMin);
    const tClamped = Math.max(0, t);
    const l1 = s[Math.max(0, tClamped - 1)];
    const l2 = s[Math.max(0, tClamped - 2)];
    return [lon, lat, Math.min(t, 12) / 10, l1, l2];
  }

  global.LSTModels = {
    TRAIN, VAL, TEST, LOOK, metrics, mean,
    train(points, normSeries, onProgress) {
      onProgress && onProgress("Hồi quy tuyến tính / đa thức theo từng điểm…");
      const xs = TRAIN.map((t) => t);
      const lin = [];
      const quad = [];
      for (let i = 0; i < normSeries.length; i++) {
        const ys = TRAIN.map((t) => normSeries[i][t]);
        lin.push(fitLine(xs, ys));
        quad.push(fitQuad(xs, ys));
      }

      onProgress && onProgress("Random Forest (Extra Trees) trên đặc trưng không gian–thời gian…");
      const tab = collectTabular(points, normSeries);
      const rf = extraTrees(tab.X, tab.y, { nTree: 22, depth: 5, minLeaf: 16, seed: 21 });

      onProgress && onProgress("XGBoost (gradient boosting stumps)…");
      const xgb = gboost(tab.X, tab.y, { rounds: 36, lr: 0.14, seed: 33 });

      onProgress && onProgress("LSTM (Elman RNN cửa sổ 4 năm)…");
      const win = collectWindows(normSeries);
      const cap = Math.min(win.seqs.length, 1400);
      const seqs = win.seqs.slice(0, cap);
      const tars = win.targets.slice(0, cap);
      const lstm = trainElman(seqs, tars, 6, 2, 0.02, 41);

      onProgress && onProgress("GRU (mạng hồi tiếp có cổng)…");
      const gru = trainGRU(seqs, tars, 5, 2, 0.015, 52);

      onProgress && onProgress("Transformer (attention trên cửa sổ 4 năm)…");
      const tr = trainAttn(seqs, tars, 3, 0.03, 63);

      function roll(i, modelId, until) {
        const hist = normSeries[i].slice(0, 7);
        const out = hist.slice();
        for (let t = 7; t <= until; t++) {
          let yhat;
          if (modelId === "lr") {
            yhat = lin[i].a + lin[i].b * t;
          } else if (modelId === "poly") {
            const c = quad[i];
            yhat = c[0] + c[1] * t + c[2] * t * t;
          } else if (modelId === "rf") {
            yhat = rf.predict(stepFeat(points[i], out, t, tab));
          } else if (modelId === "xgb") {
            yhat = xgb.predict(stepFeat(points[i], out, t, tab));
          } else {
            const seq = out.slice(t - LOOK, t);
            if (modelId === "lstm") yhat = lstm.predictSeq(seq);
            else if (modelId === "gru") yhat = gru.predictSeq(seq);
            else yhat = tr.predictSeq(seq);
          }
          yhat = Math.max(0, Math.min(1, yhat));
          out.push(yhat);
        }
        return out;
      }

      const ids = ["lr", "rf", "xgb", "lstm", "gru", "transformer"];
      const names = {
        lr: "Linear Regression",
        rf: "Random Forest",
        xgb: "XGBoost",
        lstm: "LSTM",
        gru: "GRU",
        transformer: "Transformer"
      };
      const preds = {};
      ids.forEach((id) => {
        preds[id] = [];
        for (let i = 0; i < points.length; i++) preds[id].push(roll(i, id, 12));
      });
      return { ids, names, preds };
    }
  };
})(window);
