import * as THREE from "three";

function makeSkinMat() {
  return new THREE.MeshStandardMaterial({
    color: 0xc4a07a,
    roughness: 0.68,
    metalness: 0.02,
  });
}

function makeSleeveMat() {
  return new THREE.MeshStandardMaterial({
    color: 0x3d5a80,
    roughness: 0.85,
    metalness: 0.05,
  });
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

  // 前臂：更长，近端靠近画面下沿（朝向身体），远端接手掌
  const forearm = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.048, 0.26, 6, 12),
    sleeve
  );
  forearm.name = "forearm";
  forearm.rotation.x = Math.PI / 2;
  forearm.position.set(0, 0.01, 0.06);
  forearm.castShadow = true;
  root.add(forearm);

  // 肘部附近加一节，强化「手臂」而不是短袖口
  const upper = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.055, 0.1, 6, 12),
    sleeve
  );
  upper.name = "upperArm";
  upper.rotation.x = Math.PI / 2;
  // 沿前臂轴向退后，不要横向偏移（否则易出画）
  upper.position.set(0, 0.015, 0.14);
  root.add(upper);

  const wrist = new THREE.Mesh(
    new THREE.CylinderGeometry(0.034, 0.04, 0.05, 12),
    skin
  );
  wrist.rotation.x = Math.PI / 2;
  wrist.position.set(0, 0.0, -0.12);
  root.add(wrist);

  const hand = new THREE.Group();
  hand.name = "hand";
  hand.position.set(0, 0.0, -0.22);
  // 补偿前臂下倾，让掌心朝桌、手指朝杯子方向（静止时看起来是张开而不是下抓）
  hand.rotation.x = 0.55;
  root.add(hand);

  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.028, 0.11), skin);
  palm.position.set(0, 0, -0.02);
  palm.castShadow = true;
  hand.add(palm);

  const fingers = new THREE.Group();
  fingers.name = "fingers";
  hand.add(fingers);

  // 四指：近节 + 远节；默认伸直张开，抓握时才弯曲
  const fingerXs = [-0.034, -0.012, 0.012, 0.034];
  fingerXs.forEach((x, i) => {
    const digit = new THREE.Group();
    digit.name = `digit${i}`;
    digit.position.set(x, 0.006, -0.075);
    digit.rotation.x = -0.08; // 微微上翘，避免静止像抓握
    const prox = new THREE.Mesh(new THREE.BoxGeometry(0.017, 0.015, 0.048), skin);
    prox.name = "prox";
    prox.position.z = -0.022;
    digit.add(prox);
    const dist = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.013, 0.04), skin);
    dist.name = "dist";
    dist.position.z = -0.058;
    dist.rotation.x = -0.05;
    digit.add(dist);
    fingers.add(digit);
  });

  // 拇指：自然张开在掌侧
  const thumb = new THREE.Group();
  thumb.name = "thumb";
  thumb.position.set(sign * 0.05, 0.0, -0.015);
  thumb.rotation.set(0.15, sign * -0.85, sign * 0.45);
  const tProx = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.016, 0.04), skin);
  tProx.name = "prox";
  tProx.position.z = -0.015;
  thumb.add(tProx);
  const tDist = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.014, 0.032), skin);
  tDist.name = "dist";
  tDist.position.z = -0.048;
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

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xb8c4ce);
    this.scene.fog = new THREE.Fog(0xb8c4ce, 4, 12);

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
    const hemi = new THREE.HemisphereLight(0xf0f4ff, 0x6a5a48, 0.85);
    this.scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.15);
    sun.position.set(2.2, 4.5, 1.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    this.scene.add(sun);
    const fill = new THREE.PointLight(0xffe6c8, 0.45, 8);
    fill.position.set(-0.4, 1.6, 0.6);
    this.scene.add(fill);
  }

  _buildRoom() {
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x8b6b4a,
      roughness: 0.85,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    this._floor = floor;

    const wallMat = new THREE.MeshStandardMaterial({
      color: 0xe8ddd0,
      roughness: 0.95,
    });
    const back = new THREE.Mesh(new THREE.PlaneGeometry(8, 3.2), wallMat);
    back.position.set(0, 1.6, -2.4);
    this.scene.add(back);
    this._wall = back;

    const deskMat = new THREE.MeshStandardMaterial({
      color: 0x6b4a32,
      roughness: 0.55,
    });
    const desk = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.07, 0.8), deskMat);
    desk.position.set(0, 0.95, -0.55);
    desk.castShadow = true;
    desk.receiveShadow = true;
    this.scene.add(desk);
    this._desk = desk;

    const legGeo = new THREE.BoxGeometry(0.06, 0.95, 0.06);
    const legs = [
      [-0.6, 0.475, -0.85],
      [0.6, 0.475, -0.85],
      [-0.6, 0.475, -0.25],
      [0.6, 0.475, -0.25],
    ];
    this._legs = [];
    for (const [x, y, z] of legs) {
      const leg = new THREE.Mesh(legGeo, deskMat);
      leg.position.set(x, y, z);
      leg.castShadow = true;
      this.scene.add(leg);
      this._legs.push(leg);
    }
  }

  _buildCup() {
    const g = new THREE.Group();
    g.name = "cup";
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xdfe8ef,
      roughness: 0.35,
      metalness: 0.05,
    });
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.04, 0.11, 24),
      bodyMat
    );
    body.castShadow = true;
    g.add(body);
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
    const mat = new THREE.MeshStandardMaterial({
      color: 0x7ec8e3,
      roughness: 0.25,
      metalness: 0.1,
      transparent: true,
      opacity: 0.92,
    });
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.042, 0.16, 20),
      mat
    );
    body.castShadow = true;
    g.add(body);
    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(0.016, 0.022, 0.05, 16),
      mat
    );
    neck.position.y = 0.1;
    g.add(neck);
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
    const mat = new THREE.MeshStandardMaterial({
      color: 0xc23b22,
      roughness: 0.45,
      metalness: 0.02,
    });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.048, 20, 16), mat);
    body.scale.set(1, 0.92, 1);
    body.castShadow = true;
    g.add(body);
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.004, 0.005, 0.03, 8),
      new THREE.MeshStandardMaterial({ color: 0x3a2a1a, roughness: 0.9 })
    );
    stem.position.y = 0.05;
    g.add(stem);
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
        wall: 0xe8ddd0,
        floor: 0x8b6b4a,
        desk: 0x6b4a32,
      },
      hospital_desk: {
        bg: 0xd5e4ec,
        wall: 0xf2f7fa,
        floor: 0xc5d0d8,
        desk: 0xdfe6ec,
      },
      school_desk: {
        bg: 0xd9cbb8,
        wall: 0xf0e6d2,
        floor: 0xa89070,
        desk: 0x8b5a2b,
      },
    };
    const th = themes[sceneId] || themes.home_desk;
    this._sceneId = sceneId || "home_desk";
    this.scene.background = new THREE.Color(th.bg);
    if (this.scene.fog) this.scene.fog.color = new THREE.Color(th.bg);
    if (this._wall) this._wall.material.color.setHex(th.wall);
    if (this._floor) this._floor.material.color.setHex(th.floor);
    if (this._desk) this._desk.material.color.setHex(th.desk);
    for (const leg of this._legs || []) {
      leg.material.color.setHex(th.desk);
    }
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
    const hand = msg.hand || "none";
    const anim = stage === "mi" ? msg.anim || "none" : "none";
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
      stage === "cue"
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
    }
  }

  update() {
    const t = this.clock.getElapsedTime();

    if (this.cup.userData.highlight && !this.cup.userData.held) {
      const s = 1 + Math.sin(t * 6) * 0.03;
      this.cup.scale.setScalar(s);
    } else if (!this.cup.userData.held) {
      this.cup.scale.setScalar(1);
    }

    if (this.anim !== "none" && this.handSide !== "none") {
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
}
