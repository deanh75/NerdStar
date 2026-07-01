class CamViz extends HTMLElement {
    loadedSettings = false; // To prevent saving settings back to backend on initial load
    _prevValues = {
        'cv-rl': null,
        'cv-rw': null,
        'cv-rh': null,
        'cv-tx': null,
        'cv-ty': null,
        'cv-tz': null,
        'cv-yaw': null,
        'cv-pitch': null,
        'cv-roll': null,
        'cv-fov': null
    };

    connectedCallback() {
        const self = this;
        this.style.display = 'flex';
        this.style.width = '100%';
        this.style.height = 'auto';

        this.innerHTML = `
            <!-- Panel -->
            <aside class="panel">

            <div class="sec">
                <div class="sec-h">Robot</div>
                <div class="field">
                    <div class="lbl">Length X<span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-rl" value="0.86" step="0.01">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Width Y<span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-rw" value="0.86" step="0.01">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Height Z <span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-rh" value="0.25" step="0.01">
                    </div>
                </div>
            </div>

            <div class="sec">
                <div class="sec-h">Cam Offset</div>
                <div class="field">
                    <div class="lbl">Fwd X <span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-tz" value="0.30" step="0.005">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Right Y <span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-tx" value="0" step="0.005">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Up Z <span>m</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-ty" value="0.05" step="0.005">
                    </div>
                </div>
            </div>

            <div class="sec">
                <div class="sec-h">Rotation</div>
                <div class="field">
                    <div class="lbl">Yaw <span>Deg</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-yaw" value="0" step="0.5">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Pitch <span>Deg</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-pitch" value="0" step="0.5">
                    </div>
                </div>
                <div class="field">
                    <div class="lbl">Roll <span>Deg</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-roll" value="0" step="0.5">
                    </div>
                </div>
            </div>

            <div class="sec">
                <div class="sec-h">Camera FOV</div>
                <div class="field">
                    <div class="lbl">Horiz <span>Deg</span></div>
                    <div class="ctrl">
                        <input type="number" id="cv-fov" value="70" step="1">
                    </div>
                </div>
            </div>

            </aside>

            <!-- 3D View -->
            <div class="view-wrap">
                <div class="view" id="cv-vp">
                    <canvas id="cv-canvas" tabindex="0"></canvas>

                    <div class="ov ov-tr" id="cv-readout">
                        X: 0.000 &nbsp;Y: 0.050 &nbsp;Z: 0.300<br>
                        Yaw: 0.0° &nbsp;Pitch: 0.0°  &nbsp;Roll: 0.0°
                    </div>
                    <div class="ov ov-br">LMB·ORBIT &nbsp;·&nbsp; RMB·PAN &nbsp;·&nbsp; SCROLL·ZOOM</div>
                    <div class="ov ov-bl">
                        <div class="ax"><div class="ax-bar" style="background:#4488ff"></div>X FWD</div>
                        <div class="ax"><div class="ax-bar" style="background:#ff4444"></div>Y RIGHT</div>
                        <div class="ax"><div class="ax-bar" style="background:#44ee44"></div>Z UP</div>
                    </div>
                </div>
            </div>
        `;

        (() => {
            'use strict';

            function loadThree(cb) {
                if (window.THREE) { cb(window.THREE); return; }
                const s = document.createElement('script');
                s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
                s.onload = () => cb(window.THREE);
                document.head.appendChild(s);
            }

            loadThree((T) => {
                const canvas = document.getElementById('cv-canvas');
                const vpEl = document.getElementById('cv-vp');
                const readout = document.getElementById('cv-readout');
                const NEON = 0x39FF14;

                /* Renderer */
                const renderer = new T.WebGLRenderer({ canvas, antialias: true });
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

                const scene = new T.Scene();
                scene.background = new T.Color(0x000000);

                /* Orbit camera */
                const vcam = new T.PerspectiveCamera(50, 1, 0.01, 60);
                const orb  = { theta: 0.55, phi: 1.0, r: 3.2, px: 0, py: 0.3 };
                let drag = null;

                function applyOrb() {
                    const sp = Math.sin(orb.phi), cp = Math.cos(orb.phi);
                    const st = Math.sin(orb.theta), ct = Math.cos(orb.theta);
                    vcam.position.set(orb.px + orb.r*sp*st, orb.py + orb.r*cp, orb.r*sp*ct);
                    vcam.lookAt(orb.px, orb.py, 0);
                }
                applyOrb();

                canvas.addEventListener('mousedown', e => {
                    e.preventDefault();
                    drag = { type: e.button === 2 ? 'pan' : 'orbit', x: e.clientX, y: e.clientY };
                    canvas.focus();
                });
                canvas.addEventListener('contextmenu', e => e.preventDefault());

                window.addEventListener('mousemove', e => {
                    if (!drag) return;
                    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
                    drag.x = e.clientX; drag.y = e.clientY;
                    if (drag.type === 'orbit') {
                    orb.theta -= dx * 0.007;
                    orb.phi = Math.max(0.06, Math.min(Math.PI - 0.06, orb.phi + dy * 0.007));
                    } else {
                    orb.px -= dx * orb.r * 0.001;
                    orb.py += dy * orb.r * 0.001;
                    }
                    applyOrb();
                });
                window.addEventListener('mouseup', () => drag = null);

                canvas.addEventListener('wheel', e => {
                    orb.r = Math.max(0.4, Math.min(14, orb.r * Math.pow(1.001, e.deltaY)));
                    applyOrb();
                    e.preventDefault();
                }, { passive: false });

                /* Touch */
                let prevT = [];
                canvas.addEventListener('touchstart', e => { prevT = [...e.touches]; e.preventDefault(); }, { passive: false });
                canvas.addEventListener('touchmove', e => {
                    const t = [...e.touches];
                    if (t.length === 1 && prevT.length) {
                    orb.theta -= (t[0].clientX - prevT[0].clientX) * 0.007;
                    orb.phi = Math.max(0.06, Math.min(Math.PI - 0.06, orb.phi + (t[0].clientY - prevT[0].clientY) * 0.007));
                    } else if (t.length === 2 && prevT.length === 2) {
                    const p = Math.hypot(prevT[0].clientX - prevT[1].clientX, prevT[0].clientY - prevT[1].clientY);
                    const c = Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
                    orb.r = Math.max(0.4, Math.min(14, orb.r * (p / c)));
                    }
                    applyOrb(); prevT = t; e.preventDefault();
                }, { passive: false });

                /* Lights */
                scene.add(new T.AmbientLight(0x0d1a0d, 3.5));
                const sun = new T.DirectionalLight(0x39FF14, 0.45);
                sun.position.set(2, 5, 3); scene.add(sun);

                /* Grid */
                scene.add(new T.GridHelper(10, 40, 0x0a2203, 0x020a02));

                /* Groups */
                const RG = new T.Group(), CG = new T.Group(), LG = new T.Group();
                scene.add(RG, CG, LG);

                /* Utilities */
                function clearG(g) {
                    for (let i = g.children.length - 1; i >= 0; i--) {
                    const c = g.children[i];
                    c.geometry && c.geometry.dispose();
                    c.material && [].concat(c.material).forEach(m => m.dispose());
                    g.remove(c);
                    }
                }

                function lm(color, opacity) {
                    return new T.LineBasicMaterial({ color, transparent: opacity < 1, opacity: opacity ?? 1 });
                }

                function dashedLine(x1, y1, z1, x2, y2, z2, color, op) {
                    const geo = new T.BufferGeometry().setFromPoints([
                    new T.Vector3(x1,y1,z1), new T.Vector3(x2,y2,z2)
                    ]);
                    const mat = new T.LineDashedMaterial({
                    color, transparent: true, opacity: op ?? 0.5, dashSize: 0.04, gapSize: 0.025
                    });
                    const l = new T.Line(geo, mat);
                    l.computeLineDistances();
                    return l;
                }

                /* Robot */
                function buildRobot(rw, rh, rl) {
                    const geo = new T.BoxGeometry(rw, rh, rl);

                    // Semi-transparent fill
                    const body = new T.Mesh(geo,
                    new T.MeshPhongMaterial({ color: 0x010d01, transparent: true, opacity: 0.82, shininess: 50 })
                    );
                    body.position.y = rh / 2;
                    RG.add(body);

                    // Neon edge wireframe
                    const edges = new T.LineSegments(new T.EdgesGeometry(geo), lm(NEON, 1));
                    edges.position.y = rh / 2;
                    RG.add(edges);

                    // Bumper outline (slightly larger)
                    const bEdges = new T.LineSegments(
                    new T.EdgesGeometry(new T.BoxGeometry(rw + 0.0889, 0.1143, rl + 0.0889)),
                    lm(0x1a6608, 0.6)
                    );
                    bEdges.position.y = 0.038;
                    RG.add(bEdges);

                    // Top center crosshair (X=red, Z=blue)
                    const cl = Math.min(rw, rl) * 0.27;
                    RG.add(new T.Line(
                    new T.BufferGeometry().setFromPoints([new T.Vector3(-cl, rh+0.004, 0), new T.Vector3(cl, rh+0.004, 0)]),
                    lm(0xff3333, 0.45)
                    ));
                    RG.add(new T.Line(
                    new T.BufferGeometry().setFromPoints([new T.Vector3(0, rh+0.004, -cl), new T.Vector3(0, rh+0.004, cl)]),
                    lm(0x4488ff, 0.45)
                    ));
                }

                /* Camera module */
                function buildCam() {
                    // Camera faces +Z
                    const bGeo = new T.BoxGeometry(0.085, 0.056, 0.044);
                    CG.add(new T.Mesh(bGeo, new T.MeshPhongMaterial({ color: 0x0a1e0a, shininess: 90 })));
                    CG.add(new T.LineSegments(new T.EdgesGeometry(bGeo), lm(NEON, 1)));

                    // Lens
                    const lens = new T.Mesh(
                    new T.CylinderGeometry(0.015, 0.020, 0.024, 14),
                    new T.MeshPhongMaterial({ color: NEON, emissive: 0x0d3a05, transparent: true, opacity: 0.88 })
                    );
                    lens.rotation.x = Math.PI / 2;
                    lens.position.z = 0.034;
                    CG.add(lens);

                    // Lens glow
                    const pl = new T.PointLight(NEON, 0.8, 0.8);
                    pl.position.set(0, 0, 0.06);
                    CG.add(pl);
                }

                /* FOV Frustum */
                function buildFrustum(hFovDeg) {
                    const hf = (hFovDeg * Math.PI / 180) / 2;
                    const vf = Math.atan(Math.tan(hf) / 1.7778);
                    const n = 0.06, f = 2.0;
                    const fW = Math.tan(hf)*f, fH = Math.tan(vf)*f;
                    const nW = Math.tan(hf)*n, nH = Math.tan(vf)*n;

                    const pts = [
                    [-nW,-nH,n],[nW,-nH,n],[nW,nH,n],[-nW,nH,n],
                    [-fW,-fH,f],[fW,-fH,f],[fW,fH,f],[-fW,fH,f],
                    [0,0,0]
                    ];
                    const segs = [
                    0,1, 1,2, 2,3, 3,0,
                    4,5, 5,6, 6,7, 7,4,
                    0,4, 1,5, 2,6, 3,7,
                    8,0, 8,1, 8,2, 8,3
                    ];

                    const verts = [];
                    segs.forEach(i => { const p = pts[i]; verts.push(p[0], p[1], p[2]); });
                    const geo = new T.BufferGeometry();
                    geo.setAttribute('position', new T.Float32BufferAttribute(verts, 3));
                    CG.add(new T.LineSegments(geo, lm(NEON, 0.55)));

                    // Far plane fill
                    const fp = new T.Mesh(
                    new T.PlaneGeometry(fW * 2, fH * 2),
                    new T.MeshBasicMaterial({ color: NEON, transparent: true, opacity: 0.04, side: T.DoubleSide })
                    );
                    fp.position.z = f;
                    CG.add(fp);

                    // Far-plane corner brackets
                    [[-fW,-fH],[fW,-fH],[fW,fH],[-fW,fH]].forEach(([cx,cy]) => {
                    const mk = 0.065;
                    CG.add(new T.Line(
                        new T.BufferGeometry().setFromPoints([
                        new T.Vector3(cx + mk*(cx<0?1:-1), cy, f),
                        new T.Vector3(cx, cy, f),
                        new T.Vector3(cx, cy + mk*(cy<0?1:-1), f)
                        ]),
                        lm(NEON, 0.65)
                    ));
                    });
                }

                /* Read input value */
                const gv = id => parseFloat(document.getElementById(id).value);

                /* Bind inputs */
                ['rw','rl','rh','tx','ty','tz','pitch','yaw','roll','fov'].forEach(k => {
                    const el = document.getElementById('cv-' + k);
                    if (!el) return;
                    el.addEventListener('input', update);
                });

                /* Main update */
                function update() {
                    const rw    = gv('cv-rw'),  rl   = gv('cv-rl'),   rh = gv('cv-rh');
                    const tx    = gv('cv-tx'),  ty   = gv('cv-ty'),   tz = gv('cv-tz');
                    const pitch = gv('cv-pitch'), yaw = gv('cv-yaw'), roll = gv('cv-roll');
                    const fov   = gv('cv-fov');

                    // Camera world position:
                    // X,Z from robot horizontal center
                    // Y from TOP of robot  (ty=0 → flush with top surface)
                    const cwx = tx, cwy = rh + ty, cwz = tz;

                    clearG(RG); clearG(CG); clearG(LG);

                    // Robot
                    buildRobot(rw, rh, rl);

                    // Camera group position + rotation (camera faces +Z)
                    CG.position.set(cwx, cwy, cwz);
                    CG.rotation.order = 'YXZ';
                    CG.rotation.y = yaw   * Math.PI / 180;   // Yaw+ Left
                    CG.rotation.x = pitch * Math.PI / 180;   // Pitch+ Up
                    CG.rotation.z =  roll  * Math.PI / 180;   // Roll+ CCW from front
                    buildCam();
                    buildFrustum(fov);

                    // World axes at origin
                    LG.add(new T.AxesHelper(0.28));

                    // Dashed line from robot top-center → camera
                    LG.add(dashedLine(0, rh, 0, cwx, cwy, cwz, NEON, 0.7));

                    // Vertical guide: ground → camera
                    if (cwy > 0.01) LG.add(dashedLine(cwx, 0, cwz, cwx, cwy, cwz, 0x0a2203, 0.45));

                    // Ground dot at camera XZ projection
                    const dot = new T.Mesh(
                        new T.CircleGeometry(0.035, 16),
                        new T.MeshBasicMaterial({ color: NEON, transparent: true, opacity: 0.22 })
                    );
                    dot.rotation.x = -Math.PI / 2;
                    dot.position.set(cwx, 0.001, cwz);
                    LG.add(dot);

                    // Readout
                    const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(3);
                    readout.innerHTML =
                    `X:${fmt(tz)} &nbsp;Y:${fmt(tx)} &nbsp;Z:${fmt(ty)}<br>` +
                    `Yaw:${yaw.toFixed(1)}° &nbsp;Pitch:${pitch.toFixed(1)}° &nbsp;Roll:${roll.toFixed(1)}°`;

                    // Save only changed settings
                    saveChangedSettings({
                        'cv-rl': rl,
                        'cv-rw': rw,
                        'cv-rh': rh,
                        'cv-tx': tx,
                        'cv-ty': ty,
                        'cv-tz': tz,
                        'cv-yaw': yaw,
                        'cv-pitch': pitch,
                        'cv-roll': roll,
                        'cv-fov': fov
                    });
                }

                // Save only changed settings to appropriate endpoints
                function saveChangedSettings(currentValues) {
                    const changes = [];

                    // Check each value for changes
                    for (const [id, currentValue] of Object.entries(currentValues)) {
                        if (self._prevValues[id] !== currentValue) {
                            self._prevValues[id] = currentValue; // Update the stored value
                            changes.push({ id, value: currentValue });
                        }
                    }

                    if (!self.loadedSettings) return; // Don't save on initial load

                    // If there are changes, save them
                    if (changes.length > 0) {
                        const cameraIndex = document.getElementById('cameraDropdown')?.value || null; // Get selected camera index if available

                        changes.forEach(change => {
                            const mapping = {
                                // Robot settings -> /set_local_settings (from self.local_config in get_cviz_settings)
                                'cv-rl': { endpoint: '/set_local_settings', key: 'robot_size_x' },
                                'cv-rw': { endpoint: '/set_local_settings', key: 'robot_size_y' },
                                'cv-rh': { endpoint: '/set_local_settings', key: 'robot_size_z' },
                                // Cviz settings -> /set_cviz_setting (from config in get_cviz_settings)
                                'cv-tx': { endpoint: '/set_cviz_setting', key: 'y' },
                                'cv-ty': { endpoint: '/set_cviz_setting', key: 'z' },
                                'cv-tz': { endpoint: '/set_cviz_setting', key: 'x' },
                                'cv-yaw': { endpoint: '/set_cviz_setting', key: 'yaw_degrees' },
                                'cv-pitch': { endpoint: '/set_cviz_setting', key: 'pitch_degrees' },
                                'cv-roll': { endpoint: '/set_cviz_setting', key: 'roll_degrees' },
                                // Camera FOV is a direct camera setting
                                'cv-fov': { endpoint: '/set_camera_setting', key: 'camera_horiz_fov' }
                            };

                            const map = mapping[change.id];
                            if (map) {
                                const payload = map.endpoint === '/set_local_settings'
                                    ? { key: map.key, value: change.value }
                                    : { index: cameraIndex, key: map.key, value: change.value };

                                fetch(map.endpoint, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(payload)
                                }).catch(console.error);
                            }
                        });
                    }
                };

                /* Resize */
                function resize() {
                    const w = vpEl.clientWidth;
                    const h = vpEl.clientHeight;
                    if (!w || !h) return;
                    renderer.setSize(w, h, false);
                    vcam.aspect = w / h;
                    vcam.updateProjectionMatrix();
                }
                if (window.ResizeObserver) {
                    let rafId = null;
                    new ResizeObserver(() => {
                        if (rafId) cancelAnimationFrame(rafId);
                        rafId = requestAnimationFrame(() => {
                            resize();
                            rafId = null;
                        });
                    }).observe(vpEl);
                } else {
                    window.addEventListener('resize', resize);
                }
                resize();

                /* First build */
                update();

                /* Render loop */
                (function loop() { requestAnimationFrame(loop); renderer.render(scene, vcam); })();
            });
        })();
    }

    // Public method to update component settings from outside
    updateFromSettings(settings) {
        // Map settings to component input IDs
        const settingMap = {
            // Robot dimensions
            'length_x': 'cv-rl',
            'width_y': 'cv-rw',
            'height_z': 'cv-rh',

            // Camera offset
            'fwd_x': 'cv-tz',
            'right_y': 'cv-tx',
            'up_z': 'cv-ty',

            // Rotation
            'yaw': 'cv-yaw',
            'pitch': 'cv-pitch',
            'roll': 'cv-roll',

            // Camera FOV
            'horiz_fov': 'cv-fov'
        };

        // Update each setting if provided
        for (const [key, numberId] of Object.entries(settingMap)) {
            if (settings[key] !== undefined && settings[key] !== null) {
                const numberInput = document.getElementById(numberId);
                if (numberInput) {
                    if (settings[key]?.toString().split('.')[1]?.length > 3) {
                        settings[key] = parseFloat(settings[key].toFixed(3));
                    }
                    numberInput.value = settings[key];
                }
            }
        }

        // Update _prevValues to match current settings to prevent false change detection
        for (const [key, numberId] of Object.entries(settingMap)) {
            if (settings[key] !== undefined && settings[key] !== null) {
                if (settings[key]?.toString().split('.')[1]?.length > 3) {
                    settings[key] = parseFloat(settings[key].toFixed(3));
                }
                this._prevValues[numberId] = settings[key].toFixed(3);
            }
        }

        // Trigger update to refresh the visualization
        this.querySelector('.view-wrap') && this.updateBindings_();

        this.loadedSettings = true; // Allow saving changes after initial load
    }

    // Helper method to trigger update (since update() is private in the IIFE)
    updateBindings_() {
        // We need to call the update function from the IIFE
        // Since we can't access it directly, we'll trigger input events on all number inputs
        const inputs = this.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.dispatchEvent(new Event('input'));
        });
    }
}

customElements.define('cam-viz', CamViz);