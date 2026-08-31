import * as THREE from "three";

/* ------------------------------------------------------------------ */
/* 程序化纹理（Canvas 生成，保持完全离线可用，不依赖外部素材）            */
/* ------------------------------------------------------------------ */

function _canvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")];
}

/** 通用噪点叠加：让纯色材质有微观粗糙变化 */
function _noise(ctx, w, h, alpha, dark = true) {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random() - 0.5) * 255 * alpha;
    d[i] = Math.min(255, Math.max(0, d[i] + (dark ? n : -n)));
    d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + (dark ? n : -n)));
    d[i + 2] = Math.min(255, Math.max(0, d[i + 2] + (dark ? n : -n)));
  }
  ctx.putImageData(img, 0, 0);
}

/** 木纹：底色 + 沿纹理方向的深浅条纹 + 细噪点 */
function woodTexture(base, stripe, repeatX = 2, repeatY = 1) {
  const [c, ctx] = _canvas(512, 512);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 90; i++) {
    const y = Math.random() * 512;
    const alpha = 0.04 + Math.random() * 0.10;
    ctx.strokeStyle = stripe;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1 + Math.random() * 3;
    ctx.beginPath();
    ctx.moveTo(0, y);
    // 轻微波动的纹路线
    for (let x = 0; x <= 512; x += 32) {
      ctx.lineTo(x, y + Math.sin(x * 0.02 + i) * 2);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  _noise(ctx, 512, 512, 0.05);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** 墙面：乳胶漆底色 + 微噪点 */
function wallTexture(base) {
  const [c, ctx] = _canvas(256, 256);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 256, 256);
  _noise(ctx, 256, 256, 0.03);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 2);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** 地板：木地板条 + 板缝 */
function plankTexture(base, seam) {
  const [c, ctx] = _canvas(512, 512);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 512, 512);
  const plankH = 64;
  for (let row = 0; row < 512 / plankH; row++) {
    const offset = (row % 2) * 128;
    for (let x = -128; x < 512; x += 256) {
      // 每块板轻微色差
      const v = 0.9 + Math.random() * 0.2;
      ctx.fillStyle = _shade(base, v);
      ctx.fillRect(x + offset, row * plankH + 1, 256 - 2, plankH - 2);
    }
    ctx.fillStyle = seam;
    ctx.fillRect(0, row * plankH, 512, 1.5);
  }
  // 木纹细线
  ctx.globalAlpha = 0.08;
  ctx.strokeStyle = seam;
  for (let i = 0; i < 160; i++) {
    const y = Math.random() * 512;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y + (Math.random() - 0.5) * 6);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  _noise(ctx, 512, 512, 0.04);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** 地毯：底色 + 边框 + 织物噪点 */
function rugTexture(base, border) {
  const [c, ctx] = _canvas(256, 256);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = border;
  ctx.lineWidth = 10;
  ctx.strokeRect(14, 14, 228, 228);
  ctx.lineWidth = 3;
  ctx.strokeRect(32, 32, 192, 192);
  _noise(ctx, 256, 256, 0.09);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function _shade(hex, v) {
  const c = new THREE.Color(hex);
  c.multiplyScalar(v);
  return `#${c.getHexString()}`;
}

/**
 * 程序化环境贴图：equirect 渐变（上亮下暗）+ 一块「窗户」高光，
 * 经 PMREM 卷积后供 PBR 材质反射使用。完全离线生成。
 */
function makeEnvironment(renderer) {
  const [c, ctx] = _canvas(512, 256);
  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, "#e8f0fa");
  grad.addColorStop(0.5, "#cdd6de");
  grad.addColorStop(0.62, "#a89a86");
  grad.addColorStop(1, "#6f6152");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 256);
  // 窗户亮块（会形成高光反射，位置与场景左侧窗一致）
  ctx.fillStyle = "rgba(255,250,235,0.95)";
  ctx.fillRect(60, 52, 110, 84);
  ctx.fillStyle = "rgba(255,250,235,0.35)";
  ctx.fillRect(40, 36, 150, 116);
  // 天花板灯
  ctx.fillStyle = "rgba(255,244,220,0.7)";
  ctx.fillRect(250, 8, 60, 22);
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const env = pmrem.fromEquirectangular(tex).texture;
  pmrem.dispose();
  tex.dispose();
  return env;
}

/* ------------------------------------------------------------------ */
/* 手部：胶囊 + 关节球组成的解剖结构（替代原来的盒式手指）                */
/* 层级与命名保持不变（hand/fingers/digitN/prox/dist/thumb），           */
/* setGrasp 与抓取动画逻辑无需改动。                                    */
/* ------------------------------------------------------------------ */

function makeSkinMat() {
  return new THREE.MeshStandardMaterial({
    color: 0xd6ab8e,
    roughness: 0.52,
    metalness: 0.0,
  });
}

function makeSleeveMat() {
  return new THREE.MeshStandardMaterial({
    color: 0x3d5a80,
    roughness: 0.88,
    metalness: 0.02,
  });
}

/** 一节手指：knuckle 球 + prox 胶囊组 + 关节球 + dist 胶囊组 + 指尖球 */
function _makeDigit(skin, wProx, wDist, lenProx, lenDist) {
  const digit = new THREE.Group();

  const knuckle = new THREE.Mesh(
    new THREE.SphereGeometry(wProx * 1.12, 10, 8),
    skin
  );
  knuckle.position.z = 0;
  digit.add(knuckle);

  const prox = new THREE.Group();
  prox.name = "prox";
  const proxMesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(wProx, lenProx, 4, 10),
    skin
  );
  proxMesh.rotation.x = Math.PI / 2;
  proxMesh.position.z = -(lenProx / 2 + wProx * 0.2);
  proxMesh.castShadow = true;
  prox.add(proxMesh);
  digit.add(prox);

  const joint = new THREE.Mesh(
    new THREE.SphereGeometry(wDist * 1.15, 10, 8),
    skin
  );
  joint.position.z = -(lenProx + wProx * 0.3);
  digit.add(joint);

  const dist = new THREE.Group();
  dist.name = "dist";
  dist.position.z = -(lenProx + wProx * 0.3);
  const distMesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(wDist, lenDist, 4, 10),
    skin
  );
  distMesh.rotation.x = Math.PI / 2;
  distMesh.position.z = -(lenDist / 2 + wDist * 0.2);
  distMesh.castShadow = true;
  dist.add(distMesh);
  // 指尖略收细
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(wDist * 0.95, 10, 8),
    skin
  );
  tip.position.z = -(lenDist + wDist * 0.4);
  dist.add(tip);
  digit.add(dist);
  return digit;
}

/**
 * 第一人称：前臂 + 手掌 + 可弯曲手指（含抓握）。
 * @param {"left"|"right"} side
 */
function makeArmHand(side) {
  const root = new THREE.Group();
  root.name = side === "left" ? "armL" : "armR";
  const skin = makeSkinMat();
  const sleeve = makeSleeveMat();
  const sign = side === "left" ? -1 : 1;

  // 前臂：两段胶囊叠出自然锥度，近端更粗
  const forearm = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.048, 0.26, 6, 14),
    sleeve
  );
  forearm.name = "forearm";
  forearm.rotation.x = Math.PI / 2;
  forearm.position.set(0, 0.01, 0.06);
  forearm.castShadow = true;
  root.add(forearm);

  // 肘部一节 + 袖口收边（强化「穿着衣服的手臂」）
  const upper = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.056, 0.1, 6, 14),
    sleeve
  );
  upper.name = "upperArm";
  upper.rotation.x = Math.PI / 2;
  upper.position.set(0, 0.015, 0.14);
  root.add(upper);

  const cuff = new THREE.Mesh(
    new THREE.CylinderGeometry(0.052, 0.05, 0.022, 14),
    sleeve
  );
  cuff.rotation.x = Math.PI / 2;
  cuff.position.set(0, 0.012, 0.19);
  root.add(cuff);

  const wrist = new THREE.Mesh(
    new THREE.CylinderGeometry(0.033, 0.038, 0.05, 12),
    skin
  );
  wrist.rotation.x = Math.PI / 2;
  wrist.position.set(0, 0.0, -0.12);
  root.add(wrist);

  const hand = new THREE.Group();
  hand.name = "hand";
  hand.position.set(0, 0.0, -0.22);
  hand.rotation.x = 0.55;
  root.add(hand);

  // 手掌：主体 + 掌根 + 指基座斜面（近似手掌的梯形轮廓）
  const palm = new THREE.Group();
  palm.name = "palm";
  hand.add(palm);
  const palmMain = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.026, 0.1), skin);
  palmMain.position.z = -0.02;
  palmMain.castShadow = true;
  palm.add(palmMain);
  const heel = new THREE.Mesh(new THREE.BoxGeometry(0.078, 0.03, 0.045), skin);
  heel.position.set(0, -0.002, 0.035);
  palm.add(heel);
  // 指基座：一排小珠形成指丘
  for (let i = 0; i < 4; i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6), skin);
    m.position.set(-0.034 + i * 0.0227, 0.004, -0.072);
    palm.add(m);
  }

  const fingers = new THREE.Group();
  fingers.name = "fingers";
  hand.add(fingers);

  // 四指：由内向外略短（食指~小指）
  const fingerXs = [-0.034, -0.012, 0.012, 0.034];
  const lens = [
    [0.045, 0.036], // 食指
    [0.048, 0.039], // 中指
    [0.044, 0.035], // 无名指
    [0.038, 0.03], // 小指
  ];
  fingerXs.forEach((x, i) => {
    const wP = 0.0088 - i * 0.0004;
    const wD = wP - 0.0009;
    const digit = _makeDigit(skin, wP, wD, lens[i][0], lens[i][1]);
    digit.name = `digit${i}`;
    digit.position.set(x, 0.006, -0.075);
    digit.rotation.x = -0.08; // 微微上翘，避免静止像抓握
    fingers.add(digit);
  });

  // 拇指：两节，自然张开在掌侧
  const thumb = new THREE.Group();
  thumb.name = "thumb";
  thumb.position.set(sign * 0.05, 0.0, -0.015);
  thumb.rotation.set(0.15, sign * -0.85, sign * 0.45);
  const tKnuckle = new THREE.Mesh(
    new THREE.SphereGeometry(0.0115, 10, 8),
    skin
  );
  thumb.add(tKnuckle);
  const tProx = new THREE.Group();
  tProx.name = "prox";
  const tProxMesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.0098, 0.032, 4, 10),
    skin
  );
  tProxMesh.rotation.x = Math.PI / 2;
  tProxMesh.position.z = -0.018;
  tProxMesh.castShadow = true;
  tProx.add(tProxMesh);
  thumb.add(tProx);
  const tJoint = new THREE.Mesh(new THREE.SphereGeometry(0.0092, 8, 6), skin);
  tJoint.position.z = -0.036;
  thumb.add(tJoint);
  const tDist = new THREE.Group();
  tDist.name = "dist";
  tDist.position.z = -0.036;
  const tDistMesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.0085, 0.026, 4, 10),
    skin
  );
  tDistMesh.rotation.x = Math.PI / 2;
  tDistMesh.position.z = -0.015;
  tDist.add(tDistMesh);
  const tTip = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 6), skin);
  tTip.position.z = -0.03;
  tDist.add(tTip);
  thumb.add(tDist);
  fingers.add(thumb);

  // 休息位：靠画面内侧，避免 NDC |x|>1 被裁切；y/z 保持在竖直 FOV 内
  root.userData.rest = {
    pos: new THREE.Vector3(sign * 0.14, -0.16, -0.4),
    rot: new THREE.Euler(-0.65, sign * 0.18, sign * 0.05),
  };
  root.position.copy(root.userData.rest.pos);
  root.rotation.copy(root.userData.rest.rot);
  root.userData.grasp = 0;
  return root;
}

/** @param {THREE.Object3D} arm @param {number} amount 0..1 */
function setGrasp(arm, amount) {
  const a = Math.max(0, Math.min(1, amount));
  arm.userData.grasp = a;
  const fingers = arm.getObjectByName("fingers");
  if (!fingers) return;

  fingers.children.forEach((digit) => {
    if (!digit.name.startsWith("digit") && digit.name !== "thumb") return;
    const prox = digit.getObjectByName("prox");
    const dist = digit.getObjectByName("dist");
    if (digit.name === "thumb") {
      digit.rotation.x = 0.15 + a * 0.55;
      if (prox) prox.rotation.x = a * 0.55;
      if (dist) dist.rotation.x = a * 0.85;
    } else {
      // 从微微上翘 → 向下抓握闭合
      digit.rotation.x = -0.08 + a * 1.25;
      if (prox) prox.rotation.x = a * 0.25;
      if (dist) dist.rotation.x = -0.05 + a * 1.2;
    }
  });
}

/* ------------------------------------------------------------------ */
/* 主场景                                                              */
/* ------------------------------------------------------------------ */

export class HomeDeskScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.clock = new THREE.Clock();
    this.anim = "none";
    this.handSide = "none";
    this.animT0 = 0;
    this.animDur = 4;
    this.transition = null;
    this._stageKey = "";
    this._animProgress = 0;
    this._v2ProgressEl = null;
    this._v2IdleEl = null;
    this._v2GameLabel = 0;
    this._v2ArmLevel = 0;
    this._v2ArmLevelTarget = 0;
    this._v2LastT = 0;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // ACES 色调映射：高光过渡柔和，整体更像真实相机成像
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xb8c4ce);
    this.scene.fog = new THREE.Fog(0xb8c4ce, 5, 14);
    // 程序化环境贴图：所有 PBR 材质获得统一反射光照
    this.scene.environment = makeEnvironment(this.renderer);
    this.scene.environmentIntensity = 0.55;

    this.camera = new THREE.PerspectiveCamera(
      65,
      window.innerWidth / window.innerHeight,
      0.05,
      50
    );
    this.camRest = new THREE.Vector3(0, 1.45, 0.55);
    this.camLook = new THREE.Vector3(0, 1.05, -0.55);
    this.camera.position.copy(this.camRest);
    this.camera.lookAt(this.camLook);

    this._buildLights();
    this._buildRoom();
    this.targets = {
      cup: this._buildCup(),
      bottle: this._buildBottle(),
      apple: this._buildApple(),
    };
    for (const t of Object.values(this.targets)) {
      t.visible = false;
      this.scene.add(t);
    }
    this.cup = this.targets.cup;
    this.cup.visible = true;
    this._objectId = "cup";
    this._sceneId = "home_desk";
    this._transitionAmp = "micro";

    this.handL = makeArmHand("left");
    this.handR = makeArmHand("right");
    this.camera.add(this.handL);
    this.camera.add(this.handR);
    this.scene.add(this.camera);

    window.addEventListener("resize", () => this._onResize());
  }

  _buildLights() {
    const hemi = new THREE.HemisphereLight(0xf0f4ff, 0x6a5a48, 0.55);
    this.scene.add(hemi);
    // 主光（窗光）：从左墙窗户方向入射，与环境贴图的窗亮块一致
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.5);
    sun.position.set(-2.2, 4.5, 1.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 12;
    sun.shadow.camera.left = -3;
    sun.shadow.camera.right = 3;
    sun.shadow.camera.top = 3;
    sun.shadow.camera.bottom = -3;
    sun.shadow.radius = 4;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);
    this._sun = sun;
    const fill = new THREE.PointLight(0xffe6c8, 18, 8, 2);
    fill.position.set(-0.4, 1.6, 0.6);
    this.scene.add(fill);
  }

  _buildRoom() {
    // 地板：木地板条纹理
    this._floorMat = new THREE.MeshStandardMaterial({
      map: plankTexture("#9a7350", "#5f4128"),
      roughness: 0.72,
      metalness: 0.02,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), this._floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    this._floor = floor;

    // 地毯（桌面下，增加地面层次）
    this._rugMat = new THREE.MeshStandardMaterial({
      map: rugTexture("#8a9a8f", "#6e8377"),
      roughness: 0.95,
    });
    const rug = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 1.8), this._rugMat);
    rug.rotation.x = -Math.PI / 2;
    rug.position.set(0, 0.005, -0.35);
    rug.receiveShadow = true;
    this.scene.add(rug);
    this._rug = rug;

    // 三面墙 + 踢脚线（围合感）
    this._wallMat = new THREE.MeshStandardMaterial({
      map: wallTexture("#e8ddd0"),
      roughness: 0.92,
    });
    const back = new THREE.Mesh(new THREE.PlaneGeometry(8, 3.2), this._wallMat);
    back.position.set(0, 1.6, -2.4);
    back.receiveShadow = true;
    this.scene.add(back);
    this._wall = back;

    const left = new THREE.Mesh(new THREE.PlaneGeometry(5, 3.2), this._wallMat);
    left.rotation.y = Math.PI / 2;
    left.position.set(-2.6, 1.6, -0.6);
    left.receiveShadow = true;
    this.scene.add(left);
    const right = new THREE.Mesh(new THREE.PlaneGeometry(5, 3.2), this._wallMat);
    right.rotation.y = -Math.PI / 2;
    right.position.set(2.6, 1.6, -0.6);
    right.receiveShadow = true;
    this.scene.add(right);

    const bbMat = new THREE.MeshStandardMaterial({
      color: 0xd8cfc2,
      roughness: 0.7,
    });
    for (const [w, x, ry] of [
      [8, 0, 0],
      [5, -2.6, Math.PI / 2],
      [5, 2.6, -Math.PI / 2],
    ]) {
      const bb = new THREE.Mesh(new THREE.BoxGeometry(w, 0.1, 0.02), bbMat);
      bb.position.set(x, 0.05, ry === 0 ? -2.39 : -0.6);
      bb.rotation.y = ry;
      this.scene.add(bb);
    }

    // 左墙窗户：框 + 亮窗面（静态；窗外亮度撑起整个场景的光照逻辑）
    const win = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0xf4f1ea,
      roughness: 0.5,
    });
    const pane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.9, 1.15),
      new THREE.MeshBasicMaterial({ color: 0xfff6e0 })
    );
    pane.rotation.y = Math.PI / 2;
    win.add(pane);
    for (const [w, h, y] of [
      [1.0, 0.07, 1.85],
      [1.0, 0.07, 0.65],
    ]) {
      const f = new THREE.Mesh(new THREE.BoxGeometry(0.06, h, w), frameMat);
      f.position.set(-2.58, y, -0.6);
      win.add(f);
    }
    for (const y of [1.85, 0.65]) {
      const f = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.27, 0.07), frameMat);
      f.position.set(-2.58, 1.25, -0.6 + 0.465);
      win.add(f);
      const f2 = f.clone();
      f2.position.z = -0.6 - 0.465;
      win.add(f2);
    }
    const mid = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.15, 0.05), frameMat);
    mid.position.set(-2.58, 1.25, -0.6);
    win.add(mid);
    this.scene.add(win);

    // 后墙搁板 + 书 + 小盆栽（静态生活道具，强化「真实房间」）
    const shelfMat = new THREE.MeshStandardMaterial({
      color: 0x7a5a3e,
      roughness: 0.6,
    });
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.035, 0.22), shelfMat);
    shelf.position.set(0.9, 1.95, -2.28);
    shelf.castShadow = true;
    shelf.receiveShadow = true;
    this.scene.add(shelf);
    const bookColors = [0x8c3b2e, 0x2f5d50, 0xb08d3f, 0x4a5878];
    bookColors.forEach((col, i) => {
      const b = new THREE.Mesh(
        new THREE.BoxGeometry(0.035, 0.24 - (i % 2) * 0.03, 0.16),
        new THREE.MeshStandardMaterial({ color: col, roughness: 0.8 })
      );
      b.position.set(0.42 + i * 0.05, 2.09, -2.28);
      b.rotation.z = (i - 1.5) * 0.03;
      b.castShadow = true;
      this.scene.add(b);
    });
    const pot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.04, 0.08, 12),
      new THREE.MeshStandardMaterial({ color: 0xa65b3f, roughness: 0.85 })
    );
    pot.position.set(1.15, 2.01, -2.28);
    pot.castShadow = true;
    this.scene.add(pot);
    const leafMat = new THREE.MeshStandardMaterial({
      color: 0x3f7a44,
      roughness: 0.8,
    });
    for (let i = 0; i < 4; i++) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 8), leafMat);
      s.position.set(
        1.15 + (Math.random() - 0.5) * 0.06,
        2.09 + Math.random() * 0.05,
        -2.28 + (Math.random() - 0.5) * 0.06
      );
      s.castShadow = true;
      this.scene.add(s);
    }

    // 桌子：木纹桌面
    this._deskMat = new THREE.MeshStandardMaterial({
      map: woodTexture("#8a5c3a", "#5e3a1e", 1, 1),
      roughness: 0.45,
      metalness: 0.05,
    });
    const desk = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.07, 0.8), this._deskMat);
    desk.position.set(0, 0.95, -0.55);
    desk.castShadow = true;
    desk.receiveShadow = true;
    this.scene.add(desk);
    this._desk = desk;

    // 桌上小道具：笔记本 + 笔（远离目标位，不干扰注意）
    const padMat = new THREE.MeshStandardMaterial({
      color: 0x30475e,
      roughness: 0.7,
    });
    const pad = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.012, 0.18), padMat);
    pad.position.set(-0.5, 0.99, -0.72);
    pad.rotation.y = 0.15;
    pad.castShadow = true;
    this.scene.add(pad);
    const pen = new THREE.Mesh(
      new THREE.CylinderGeometry(0.005, 0.005, 0.16, 8),
      new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.4 })
    );
    pen.rotation.set(Math.PI / 2, 0, 0.5);
    pen.position.set(-0.42, 1.0, -0.6);
    pen.castShadow = true;
    this.scene.add(pen);

    const legGeo = new THREE.BoxGeometry(0.06, 0.95, 0.06);
    const legs = [
      [-0.6, 0.475, -0.85],
      [0.6, 0.475, -0.85],
      [-0.6, 0.475, -0.25],
      [0.6, 0.475, -0.25],
    ];
    this._legs = [];
    for (const [x, y, z] of legs) {
      const leg = new THREE.Mesh(legGeo, this._deskMat);
      leg.position.set(x, y, z);
      leg.castShadow = true;
      this.scene.add(leg);
      this._legs.push(leg);
    }
  }

  _buildCup() {
    const g = new THREE.Group();
    g.name = "cup";
    const bodyMat = new THREE.MeshPhysicalMaterial({
      color: 0xeef2f5,
      roughness: 0.22,
      metalness: 0.02,
      clearcoat: 0.4,
      clearcoatRoughness: 0.25,
    });
    // 开口杯体：外壳 + 内衬 + 杯口环
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.04, 0.11, 28, 1, true),
      bodyMat
    );
    body.castShadow = true;
    g.add(body);
    const inner = new THREE.Mesh(
      new THREE.CylinderGeometry(0.041, 0.037, 0.108, 28, 1, true),
      new THREE.MeshStandardMaterial({
        color: 0xcfd6da,
        roughness: 0.35,
        side: THREE.BackSide,
      })
    );
    g.add(inner);
    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(0.044, 0.003, 10, 28),
      bodyMat
    );
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.055;
    g.add(rim);
    // 咖啡液面
    const coffee = new THREE.Mesh(
      new THREE.CircleGeometry(0.039, 24),
      new THREE.MeshStandardMaterial({
        color: 0x3a2415,
        roughness: 0.15,
      })
    );
    coffee.rotation.x = -Math.PI / 2;
    coffee.position.y = 0.04;
    g.add(coffee);
    const h1 = new THREE.Mesh(
      new THREE.TorusGeometry(0.03, 0.007, 10, 20, Math.PI),
      bodyMat
    );
    h1.rotation.set(Math.PI / 2, 0, Math.PI / 2);
    h1.position.set(0.05, 0, 0);
    g.add(h1);
    g.position.set(0, 1.055, -0.55);
    g.userData.restPos = g.position.clone();
    g.userData.highlight = false;
    g.userData.held = false;
    g.userData.away = false;
    return g;
  }

  _buildBottle() {
    const g = new THREE.Group();
    g.name = "bottle";
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x7ec8e3,
      roughness: 0.15,
      metalness: 0.05,
      transparent: true,
      opacity: 0.88,
      clearcoat: 0.6,
    });
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.042, 0.16, 24),
      mat
    );
    body.castShadow = true;
    g.add(body);
    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(0.016, 0.022, 0.05, 18),
      mat
    );
    neck.position.y = 0.1;
    g.add(neck);
    const cap = new THREE.Mesh(
      new THREE.CylinderGeometry(0.017, 0.017, 0.022, 18),
      new THREE.MeshStandardMaterial({ color: 0x2b6cb0, roughness: 0.5 })
    );
    cap.position.y = 0.135;
    cap.castShadow = true;
    g.add(cap);
    // 标签环
    const label = new THREE.Mesh(
      new THREE.CylinderGeometry(0.0362, 0.0385, 0.05, 24, 1, true),
      new THREE.MeshStandardMaterial({
        color: 0xf5f1e8,
        roughness: 0.8,
        side: THREE.DoubleSide,
      })
    );
    label.position.y = -0.01;
    g.add(label);
    g.position.set(0, 1.08, -0.55);
    g.userData.restPos = g.position.clone();
    g.userData.highlight = false;
    g.userData.held = false;
    g.userData.away = false;
    return g;
  }

  _buildApple() {
    const g = new THREE.Group();
    g.name = "apple";
    // 双色苹果纹理
    const [c, ctx] = _canvas(128, 128);
    const grad = ctx.createLinearGradient(0, 0, 128, 0);
    grad.addColorStop(0, "#c23b22");
    grad.addColorStop(0.55, "#d1553a");
    grad.addColorStop(1, "#e0b46a");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    _noise(ctx, 128, 128, 0.06);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshPhysicalMaterial({
      map: tex,
      roughness: 0.30,
      clearcoat: 0.5,
      clearcoatRoughness: 0.3,
    });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.048, 24, 18), mat);
    body.scale.set(1, 0.92, 1);
    body.castShadow = true;
    g.add(body);
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.004, 0.005, 0.03, 8),
      new THREE.MeshStandardMaterial({ color: 0x3a2a1a, roughness: 0.9 })
    );
    stem.position.y = 0.05;
    stem.rotation.z = 0.15;
    g.add(stem);
    const leaf = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 10, 6),
      new THREE.MeshStandardMaterial({ color: 0x4a7a3a, roughness: 0.8 })
    );
    leaf.scale.set(1.6, 0.3, 0.8);
    leaf.position.set(0.015, 0.055, 0);
    leaf.rotation.z = -0.4;
    g.add(leaf);
    g.position.set(0, 1.04, -0.55);
    g.userData.restPos = g.position.clone();
    g.userData.highlight = false;
    g.userData.held = false;
    g.userData.away = false;
    return g;
  }

  setObject(objectId) {
    const id = this.targets[objectId] ? objectId : "cup";
    if (id === this._objectId && this.cup === this.targets[id]) return;
    // 卸下旧目标
    this._resetCup();
    for (const [k, t] of Object.entries(this.targets)) {
      t.visible = k === id;
      if (t.parent !== this.scene) this.scene.add(t);
      t.position.copy(t.userData.restPos);
    }
    this.cup = this.targets[id];
    this._objectId = id;
  }

  setSceneTheme(sceneId) {
    const themes = {
      home_desk: {
        bg: 0xb8c4ce,
        wall: "#e8ddd0",
        floor: "#9a7350",
        desk: "#8a5c3a",
        rug: ["#8a9a8f", "#6e8377"],
        sun: 0xfff2dd,
      },
      hospital_desk: {
        bg: 0xd5e4ec,
        wall: "#f2f7fa",
        floor: "#b9c4cc",
        desk: "#cfd8e0",
        rug: ["#aebfc9", "#93a7b3"],
        sun: 0xf4faff,
      },
      school_desk: {
        bg: 0xd9cbb8,
        wall: "#f0e6d2",
        floor: "#a89070",
        desk: "#8b5a2b",
        rug: ["#b39b7a", "#96805f"],
        sun: 0xffefd2,
      },
    };
    const th = themes[sceneId] || themes.home_desk;
    this._sceneId = sceneId || "home_desk";
    this.scene.background = new THREE.Color(th.bg);
    if (this.scene.fog) this.scene.fog.color = new THREE.Color(th.bg);
    if (this._wall) this._wall.material.color.set(new THREE.Color(th.wall));
    if (this._floorMat)
      this._floorMat.color.set(new THREE.Color(th.floor));
    if (this._deskMat) this._deskMat.color.set(new THREE.Color(th.desk));
    if (this._rugMat && this._rugMat.map) {
      // 地毯纹理为整体染色（保留织物细节）
      this._rugMat.color.set(new THREE.Color(th.rug[0]));
    }
    if (this._sun) this._sun.color.set(new THREE.Color(th.sun));
  }

  setHudHighlight(on) {
    this.cup.userData.highlight = !!on;
  }

  _resetCup() {
    // 从手臂上卸下，放回桌面
    if (this.cup.parent && this.cup.parent !== this.scene) {
      this.scene.add(this.cup);
    }
    this.cup.visible = true;
    this.cup.userData.held = false;
    this.cup.userData.away = false;
    this.cup.position.copy(this.cup.userData.restPos);
    this.cup.rotation.set(0, 0, 0);
    this.cup.scale.setScalar(1);
  }

  /** 杯子挂到掌心（世界坐标对齐后改父节点） */
  _attachCupToHand(arm) {
    const hand = arm.getObjectByName("hand");
    if (!hand) return;
    if (this.cup.parent !== hand) {
      hand.attach(this.cup);
    }
    // 掌心前方略上方
    this.cup.position.set(0, 0.03, -0.02);
    this.cup.rotation.set(0.2, 0, 0);
    this.cup.visible = true;
    this.cup.userData.held = true;
    this.cup.userData.away = false;
  }

  _hideCupAway() {
    if (this.cup.parent !== this.scene) {
      this.scene.add(this.cup);
    }
    this.cup.visible = false;
    this.cup.userData.held = false;
    this.cup.userData.away = true;
    this.cup.position.copy(this.cup.userData.restPos);
    this.cup.rotation.set(0, 0, 0);
  }

  _ease(t) {
    const x = Math.max(0, Math.min(1, t));
    return x * x * (3 - 2 * x);
  }

  /**
   * 第一人称可读的抓取位姿（相机局部，不依赖易漂移的 worldToLocal）。
   * -Z 朝前，+Y 朝上；比休息位明显更靠画面中心、更伸向桌面。
   */
  _poseReach(side) {
    const sign = side === "left" ? -1 : 1;
    return {
      pos: new THREE.Vector3(sign * 0.05, -0.01, -0.98),
      rot: new THREE.Euler(-0.12, sign * 0.04, sign * 0.02),
    };
  }

  /** 抬杯：抬高并开始侧移 */
  _poseLift(side) {
    const sign = side === "left" ? -1 : 1;
    return {
      pos: new THREE.Vector3(sign * 0.28, 0.14, -0.62),
      rot: new THREE.Euler(-0.35, sign * 0.32, sign * 0.12),
    };
  }

  /** 取走：移出画面侧下方 */
  _poseAway(side) {
    const sign = side === "left" ? -1 : 1;
    return {
      pos: new THREE.Vector3(sign * 0.62, -0.22, -0.32),
      rot: new THREE.Euler(-0.75, sign * 0.45, sign * 0.2),
    };
  }

  /** 弱辅助前伸（不到抓取那么远） */
  _poseReachSoft(side) {
    const sign = side === "left" ? -1 : 1;
    return {
      pos: new THREE.Vector3(sign * 0.08, -0.06, -0.78),
      rot: new THREE.Euler(-0.28, sign * 0.1, sign * 0.04),
    };
  }

  /**
   * @param {{ anim?: string, hand?: string, duration_s?: number, stage?: string, trial_id?: any }} msg
   */
  applyStage(msg) {
    const stage = msg.stage || "idle";
    const phase = msg.phase || "";
    let anim = stage === "mi" ? msg.anim || "none" : "none";
    if (stage === "mi" && phase === "acquire" && anim !== "none") {
      console.error("[scene] acquire MI 段 anim 须为 none，已强制", { anim, trial_id: msg.trial_id });
      anim = "none";
    }
    const hand = msg.hand || "none";
    const key = `${stage}|${hand}|${anim}|${msg.trial_id ?? ""}`;
    const sameMi = stage === "mi" && anim !== "none" && key === this._stageKey;

    if (msg.object) this.setObject(msg.object);
    if (msg.scene) this.setSceneTheme(msg.scene);
    this._transitionAmp = msg.transition_amp || "micro";

    this.handSide = hand;
    this.anim = anim;
    this.animDur = Math.max(0.5, Number(msg.duration_s) || 4);
    if (!sameMi) {
      this.animT0 = this.clock.getElapsedTime();
      this._stageKey = key;
    }
    this.setHudHighlight(stage === "cue");

    if (stage !== "mi") {
      this._resetHands();
      this._resetCup();
    } else if (anim === "none") {
      this._resetHands();
      this._resetCup();
    } else if (this.anim === "full_grasp" || this.anim === "reach") {
      if (!this.cup.userData.held) this._resetCup();
    } else {
      this._resetCup();
    }

    const ampMap = { micro: 0.04, swap: 0.1, scene: 0.18 };
    const amp = ampMap[this._transitionAmp] || 0.04;

    if (stage === "transition") {
      this.transition = {
        t0: this.clock.getElapsedTime(),
        dur: this.animDur,
        from: this.camera.position.clone(),
        to: this.camRest
          .clone()
          .add(new THREE.Vector3(amp * 1.2, 0.01 + amp * 0.15, -0.03 - amp)),
      };
    } else if (stage === "fixation") {
      this.transition = {
        t0: this.clock.getElapsedTime(),
        dur: this.animDur,
        from: this.camera.position.clone(),
        to: this.camRest.clone().lerp(new THREE.Vector3(0, 1.42, 0.42), 0.35),
      };
    } else if (
      stage === "rest" ||
      stage === "mi" ||
      stage === "post_mi_hold" ||
      stage === "cue" ||
      stage === "settle" ||
      stage === "session_split"
    ) {
      this.transition = null;
      this.camera.position.copy(this.camRest);
      this.camera.lookAt(this.camLook);
    }
  }

  _resetHands() {
    for (const h of [this.handL, this.handR]) {
      h.position.copy(h.userData.rest.pos);
      h.rotation.copy(h.userData.rest.rot);
      setGrasp(h, 0);
    }
  }

  _setArmPose(arm, pose) {
    arm.position.copy(pose.pos);
    arm.rotation.copy(pose.rot);
  }

  _lerpArmPose(arm, a, b, t) {
    const e = this._ease(t);
    arm.position.lerpVectors(a.pos, b.pos, e);
    arm.rotation.set(
      THREE.MathUtils.lerp(a.rot.x, b.rot.x, e),
      THREE.MathUtils.lerp(a.rot.y, b.rot.y, e),
      THREE.MathUtils.lerp(a.rot.z, b.rot.z, e)
    );
  }

  _animateHand(arm, mode, u, side) {
    const rest = {
      pos: arm.userData.rest.pos,
      rot: arm.userData.rest.rot,
    };

    if (mode === "full_grasp") {
      // 伸手 → 抓握 → 抬起 → 侧移取走 → 空手复位（动作在画面内要「一眼能看懂」）
      const reach = this._poseReach(side);
      const lift = this._poseLift(side);
      const away = this._poseAway(side);

      if (u < 0.22) {
        const t = u / 0.22;
        this._lerpArmPose(arm, rest, reach, t);
        setGrasp(arm, 0);
        this._resetCup();
      } else if (u < 0.38) {
        this._setArmPose(arm, reach);
        setGrasp(arm, (u - 0.22) / 0.16);
        this._attachCupToHand(arm);
      } else if (u < 0.55) {
        const t = (u - 0.38) / 0.17;
        this._lerpArmPose(arm, reach, lift, t);
        setGrasp(arm, 1);
        // 杯子已挂在手上，随手臂移动
      } else if (u < 0.78) {
        const t = (u - 0.55) / 0.23;
        this._lerpArmPose(arm, lift, away, t);
        setGrasp(arm, 1);
        if (t > 0.55) this._hideCupAway();
      } else {
        const t = (u - 0.78) / 0.22;
        this._lerpArmPose(arm, away, rest, t);
        setGrasp(arm, 1 - t);
        this.cup.visible = false;
        this.cup.userData.away = true;
        this.cup.userData.held = false;
        if (this.cup.parent !== this.scene) this.scene.add(this.cup);
      }
    } else if (mode === "reach") {
      const soft = this._poseReachSoft(side);
      const t = u < 0.5 ? u / 0.5 : 1 - (u - 0.5) / 0.5;
      this._lerpArmPose(arm, rest, soft, Math.min(1, t));
      setGrasp(arm, 0);
      this._resetCup();
    } else if (mode === "v2_grasp") {
      // 满分：抓住杯子并略抬起（不侧移取走）
      const reach = this._poseReach(side);
      const lift = this._poseLift(side);
      if (u < 0.3) {
        this._setArmPose(arm, reach);
        setGrasp(arm, u / 0.3);
        this._attachCupToHand(arm);
      } else if (u < 0.7) {
        const t2 = (u - 0.3) / 0.4;
        this._lerpArmPose(arm, reach, lift, t2);
        setGrasp(arm, 1);
      } else {
        this._setArmPose(arm, lift);
        setGrasp(arm, 1);
      }
    }
  }

  _applyV2ArmLevel(arm, side, level01) {
    const rest = { pos: arm.userData.rest.pos, rot: arm.userData.rest.rot };
    const deep = this._poseReach(side);
    const t = Math.max(0, Math.min(1, level01));
    // 档位 0 也略微前伸，避免「完全不动」看不出启动
    const reachT = t <= 0 ? 0.06 : 0.12 + t * 0.88;
    arm.position.lerpVectors(rest.pos, deep.pos, reachT);
    arm.rotation.set(
      THREE.MathUtils.lerp(rest.rot.x, deep.rot.x, reachT),
      THREE.MathUtils.lerp(rest.rot.y, deep.rot.y, reachT),
      THREE.MathUtils.lerp(rest.rot.z, deep.rot.z, reachT)
    );
    setGrasp(arm, Math.min(0.55, t * 0.45));
    if (!this.cup.userData.held) this._resetCup();
  }

  update() {
    const t = this.clock.getElapsedTime();
    const dt = this._v2LastT > 0 ? Math.min(0.05, Math.max(0, t - this._v2LastT)) : 0.016;
    this._v2LastT = t;

    if (this.cup.userData.highlight && !this.cup.userData.held) {
      const s = 1 + Math.sin(t * 6) * 0.03;
      this.cup.scale.setScalar(s);
    } else if (!this.cup.userData.held) {
      this.cup.scale.setScalar(1);
    }

    if (this.anim === "v2_level" && this.handSide !== "none") {
      const target = Math.max(this._v2ArmLevelTarget || 0, this._v2ArmLevel || 0);
      this._v2ArmLevelTarget = target;
      const tau = 0.22;
      const k = 1 - Math.exp(-dt / tau);
      this._v2ArmLevel += (target - this._v2ArmLevel) * k;
      const hand = this.handSide === "left" ? this.handL : this.handR;
      const other = this.handSide === "left" ? this.handR : this.handL;
      other.position.copy(other.userData.rest.pos);
      other.rotation.copy(other.userData.rest.rot);
      setGrasp(other, 0);
      this._applyV2ArmLevel(hand, this.handSide, this._v2ArmLevel);
      this._animProgress = this._v2ArmLevel;
    } else if (this.anim !== "none" && this.handSide !== "none") {
      const u = Math.min(1, (t - this.animT0) / this.animDur);
      const hand = this.handSide === "left" ? this.handL : this.handR;
      const other = this.handSide === "left" ? this.handR : this.handL;
      other.position.copy(other.userData.rest.pos);
      other.rotation.copy(other.userData.rest.rot);
      setGrasp(other, 0);
      this._animateHand(hand, this.anim, u, this.handSide);
      this._animProgress = u;
    }

    if (this.transition) {
      const u = Math.min(1, (t - this.transition.t0) / this.transition.dur);
      const e = u * u * (3 - 2 * u);
      this.camera.position.lerpVectors(this.transition.from, this.transition.to, e);
      this.camera.lookAt(this.camLook);
      if (u >= 1) this.transition = null;
    }

    this.renderer.render(this.scene, this.camera);
  }

  /** 自测用：当前主动作手臂世界坐标 */
  debugArmState() {
    const hand = this.handSide === "left" ? this.handL : this.handR;
    if (!hand || this.handSide === "none") {
      return { handSide: this.handSide, anim: this.anim };
    }
    const wp = new THREE.Vector3();
    hand.getWorldPosition(wp);
    return {
      handSide: this.handSide,
      anim: this.anim,
      u: this._animProgress ?? 0,
      local: hand.position.toArray(),
      world: wp.toArray(),
      cupVisible: this.cup.visible,
      cupHeld: !!this.cup.userData.held,
      cupAway: !!this.cup.userData.away,
    };
  }

  _onResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /* ----- v2 会话挂点（v2_bridge → window.__v2scene） ----- */

  v2Fixation() {
    this.handSide = "none";
    this.anim = "none";
    this._v2ArmLevel = 0;
    this._v2ArmLevelTarget = 0;
    this._resetHands();
    this._resetCup();
    this.setHudHighlight(false);
    this._v2ProgressEl?.remove();
    this._v2ProgressEl = null;
    this._v2IdleEl?.remove();
    this._v2IdleEl = null;
  }

  v2Cue(label) {
    this._v2GameLabel = Number(label) || 0;
    this._v2ArmLevel = 0;
    this._v2ArmLevelTarget = 0;
    this._v2ProgressEl?.remove();
    this._v2ProgressEl = null;
    this._v2IdleEl?.remove();
    this._v2IdleEl = null;
    // cue 仅提示侧别与杯子高亮，不按得分伸出（得分由 judge 驱动）
    if (label === 1 || label === 2) {
      this.handSide = label === 2 ? "right" : "left";
      this.anim = "none";
      this._resetHands();
      this._resetCup();
      this.setHudHighlight(true);
      this.cup.userData.highlight = true;
    } else {
      this.v2Fixation();
    }
  }

  v2CalProgress(p) {
    let bar = this._v2ProgressEl;
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "v2-cal-progress";
      bar.style.cssText =
        "position:fixed;left:10%;right:10%;bottom:8%;height:6px;background:#3338;border-radius:3px;z-index:50";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.cssText =
        "height:100%;width:0%;background:#4ade80;border-radius:3px;transition:width .12s";
      bar.appendChild(fill);
      document.body.appendChild(bar);
      this._v2ProgressEl = bar;
    }
    const fill = bar.querySelector(".fill");
    const t = typeof p === "number" ? p : 0;
    if (fill) fill.style.width = `${Math.min(100, Math.max(0, t * 100))}%`;
  }

  v2GameLevel(level, reach, labelOpt, progress) {
    if (labelOpt === 1 || labelOpt === 2) {
      this._v2GameLabel = Number(labelOpt);
    }
    const label = this._v2GameLabel;
    if (!label || label === 0) {
      this.v2Fixation();
      return;
    }
    const side = label === 2 ? "right" : "left";
    this.handSide = side;
    this.setHudHighlight(true);
    this.cup.userData.highlight = true;
    this._v2ProgressEl?.remove();
    this._v2ProgressEl = null;

    if (reach) {
      this._v2ArmLevelTarget = 1;
      this._v2ArmLevel = Math.max(this._v2ArmLevel || 0, 0.85);
      this.anim = "v2_grasp";
      this.animT0 = this.clock.getElapsedTime();
      this.animDur = 1.0;
      return;
    }

    let target;
    if (progress != null && Number.isFinite(Number(progress))) {
      target = Math.max(0, Math.min(1, Number(progress)));
    } else {
      const lv = Math.max(0, Math.min(4, Number(level) || 0));
      target = lv / 4;
    }
    // 同试次只前进不回缩
    this._v2ArmLevelTarget = Math.max(this._v2ArmLevelTarget || 0, target);
    if (this.anim !== "v2_grasp") {
      this.anim = "v2_level";
    }
  }

  v2Iti() {
    this._v2ProgressEl?.remove();
    this._v2ProgressEl = null;
    this._v2IdleEl?.remove();
    this._v2IdleEl = null;
    this.v2Fixation();
  }

  v2Idle(text) {
    this._v2ProgressEl?.remove();
    this._v2ProgressEl = null;
    this._v2IdleEl?.remove();
    this._v2IdleEl = null;
    this.handSide = "none";
    this.anim = "none";
    this._v2ArmLevel = 0;
    this._v2ArmLevelTarget = 0;
    this._resetHands();
    this._resetCup();
    this.setHudHighlight(false);
    const title = typeof text === "object" && text ? (text.title || "") : String(text || "");
    const sub = typeof text === "object" && text ? (text.sub || "") : "";
    if (!title && !sub) return;
    const el = document.createElement("div");
    el.id = "v2-idle-overlay";
    el.style.cssText =
      "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;background:#0b1220cc;color:#e8eefc;z-index:60;pointer-events:none;text-align:center;padding:24px";
    el.innerHTML = [
      title ? `<div style="font-size:clamp(22px,4vw,36px);font-weight:600;margin-bottom:12px">${title}</div>` : "",
      sub ? `<div style="font-size:clamp(14px,2.2vw,20px);opacity:.75;max-width:640px;line-height:1.5">${sub}</div>` : "",
    ].join("");
    document.body.appendChild(el);
    this._v2IdleEl = el;
  }
}
