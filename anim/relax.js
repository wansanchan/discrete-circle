// Pure relaxation for a closed chain of edge vectors.
// The center is not stored. It is measured from the edges after the fact.
(function (root) {
  function wrap(d) {
    const t = Math.PI * 2;
    while (d > Math.PI) d -= t;
    while (d <= -Math.PI) d += t;
    return d;
  }

  function createState() {
    // A short, uneven chain. The last edge does not finish the loop,
    // and the lengths and turns are deliberately not regular.
    const edges = [
      [3.0, 0.05],
      [0.15, 0.55],
      [-1.5, 0.25],
      [-1.35, 0.55],
      [-0.2, -0.45],
      [0.15, -0.15]
    ];
    return {
      n: edges.length,
      phi: edges.map(function (e) { return Math.atan2(e[1], e[0]); }),
      L: edges.map(function (e) { return Math.hypot(e[0], e[1]); })
    };
  }

  function step(state, dt, gain) {
    const n = state.n;
    const alpha = (Math.PI * 2) / n;
    const delta = new Array(n);
    for (let k = 0; k < n; k++) {
      const prev = (k - 1 + n) % n;
      delta[k] = wrap(state.phi[k] - state.phi[prev]);
    }
    const dPhi = new Array(n);
    for (let k = 0; k < n; k++) {
      const next = (k + 1) % n;
      dPhi[k] = (delta[k] - alpha) - (delta[next] - alpha);
    }
    const kAngle = gain.even;
    const kLen = gain.even;
    for (let k = 0; k < n; k++) {
      state.phi[k] -= kAngle * dPhi[k] * dt;
      state.L[k] -= kLen * (state.L[k] - 1) * dt;
      if (state.L[k] < 0.18) state.L[k] = 0.18;
    }
    let mx = 0;
    let my = 0;
    const cs = new Array(n);
    const sn = new Array(n);
    for (let k = 0; k < n; k++) {
      cs[k] = Math.cos(state.phi[k]);
      sn[k] = Math.sin(state.phi[k]);
      mx += state.L[k] * cs[k];
      my += state.L[k] * sn[k];
    }
    mx /= n;
    my /= n;
    const blend = 1 - Math.exp(-gain.close * dt);
    for (let k = 0; k < n; k++) {
      const x = state.L[k] * cs[k] - blend * mx;
      const y = state.L[k] * sn[k] - blend * my;
      state.L[k] = Math.hypot(x, y);
      state.phi[k] = Math.atan2(y, x);
    }
  }

  function geometry(state) {
    const n = state.n;
    const pts = [];
    let x = 0;
    let y = 0;
    for (let k = 0; k < n; k++) {
      pts.push({ x: x, y: y });
      x += state.L[k] * Math.cos(state.phi[k]);
      y += state.L[k] * Math.sin(state.phi[k]);
    }
    const gap = Math.hypot(x, y);
    let cx = 0;
    let cy = 0;
    for (let k = 0; k < n; k++) {
      cx += pts[k].x;
      cy += pts[k].y;
    }
    cx /= n;
    cy /= n;
    let align = 0;
    let area2 = 0;
    const normals = [];
    for (let k = 0; k < n; k++) {
      const a = pts[k];
      const b = k + 1 < n ? pts[k + 1] : { x: x, y: y };
      area2 += a.x * b.y - b.x * a.y;
    }
    const sign = area2 >= 0 ? 1 : -1;
    for (let k = 0; k < n; k++) {
      const a = pts[k];
      const ex = state.L[k] * Math.cos(state.phi[k]);
      const ey = state.L[k] * Math.sin(state.phi[k]);
      const len = Math.hypot(ex, ey) || 1;
      const nx = sign * (-ey / len);
      const ny = sign * (ex / len);
      const mx = a.x + ex / 2;
      const my = a.y + ey / 2;
      const ux = cx - mx;
      const uy = cy - my;
      const ul = Math.hypot(ux, uy) || 1;
      align += nx * (ux / ul) + ny * (uy / ul);
      normals.push({ x: mx, y: my, nx: nx, ny: ny });
    }
    align /= n;
    const peri = state.L.reduce(function (s, v) { return s + v; }, 0);
    let area = 0;
    for (let k = 0; k < n; k++) {
      const a = pts[k];
      const ex = state.L[k] * Math.cos(state.phi[k]);
      const ey = state.L[k] * Math.sin(state.phi[k]);
      area += a.x * ey - a.y * ex;
    }
    area = Math.abs(area) / 2;
    return {
      pts: pts,
      end: { x: x, y: y },
      gap: gap,
      cx: cx,
      cy: cy,
      align: align,
      normals: normals,
      peri: peri,
      area: area,
      round: area > 0 ? (4 * Math.PI * area) / (peri * peri) : 0
    };
  }

  function splitOneEdge(state) {
    let m = 0;
    for (let k = 1; k < state.n; k++) {
      if (state.L[k] > state.L[m]) m = k;
    }
    const phi = state.phi.slice();
    const L = state.L.slice();
    const heading = phi[m];
    const half = L[m] / 2;
    phi.splice(m, 1, heading, heading);
    L.splice(m, 1, half, half);
    state.n += 1;
    state.phi = phi;
    state.L = L;
  }

  root.CircleRelax = {
    createState: createState,
    step: step,
    geometry: geometry,
    splitOneEdge: splitOneEdge
  };
})(typeof window !== "undefined" ? window : globalThis);
