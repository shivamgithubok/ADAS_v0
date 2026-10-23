import QtQuick
import QtQuick.Shapes
import QtQuick3D
import QtQuick3D.AssetUtils
import QtQuick.Effects

Item {
    id: root

    // Camera — 3rd-person chase, cluster-style (low, behind ego, looking down the road)
    property real pitchDeg: 8.0
    property real yawDeg: 0.0
    property real zoomFactor: 1.08
    property real camH: 5.0
    property real camDist: 10.5
    property real calibPitch: 0.0
    property real calibH: 1.0
    property real panX: 0.0
    property real panY: 0.0
    property bool cinematicRoad: true
    property bool showLaneLines: true
    property bool scenicView: false
    property bool showCalib: false
    property bool clusterChrome: false
    // Metric debug overlay: ego centreline, 10 m ruler ticks, per-object (x, z)
    // labels in metres. Toggle from Python via _set("debugMetric", True).
    property bool debugMetric: false
    property real debugMaxZ: 70.0
    property real diskRadiusM: 70.0
    // Parsed trafficJson rows, kept for the metric labels above. Populated by
    // applyTraffic so the labels can never disagree with what was drawn.
    property var debugRows: []
    // Background: Auto (clock) | Day | Dusk | Night | Demo
    property string envMode: "auto"
    property real clockHour: 12.0
    property string envPhase: "day"
    property string envLabel: "Env Auto"
    property color skyTopColor: "#1a2740"
    property color skyMidColor: "#3a4a62"
    property color skyBotColor: "#6b7a90"
    property color clearCol: "#0e1218"
    property color lightCol: "#fff4e8"
    property color ambientCol: "#4a5564"
    property real lightBright: 1.35
    property real sunElev: 0.65
    property real sunAzim: 18.0
    property color sunColor: "#ffe2a8"
    property real sunGlowScale: 1.0
    property bool sunIsMoon: false
    property color asphaltCol: "#101216"
    property color shoulderCol: "#0a0c10"
    property color mistCol: "#6b7a90"
    property real mistOpacity: 0.55
    property color mountainFarCol: "#3a4a5c"
    property color mountainNearCol: "#1e2834"
    property real sunPosX: 20
    property real sunPosY: 40
    property real sunPosZ: -90
    property string cipoStatus: "SAFE"
    property bool cipoVisible: false
    property real cipoX: 0
    property real cipoZ: -12
    property real cipoDist: 0
    property string ldwSide: ""
    property string fcwLevel: "OFF"

    // ---- Cluster style (Cluster.qml) ---------------------------------
    // Dark, transparent scene that sits inside the instrument-cluster bezel.
    // Colour is spent only on threat state: green / yellow / red (spec §5).
    property bool clusterStyle: false
    property bool interactive: true
    readonly property color threatGreen: "#4DFFB4"
    readonly property color threatYellow: "#FFD84D"
    readonly property color threatRed: "#FF5E7E"
    readonly property color neutralLine: "#F4F7FB"
    readonly property color offLine: "#4A4A5C"
    // Lane-line state per side: normal | caution | warning | off
    property string laneLeftState: "normal"
    property string laneRightState: "normal"
    // Gap band between ego and lead (0 none, 1 green, 2 yellow, 3 red)
    property int gapLevel: 0
    property string chevronJson: "[]"
    property int chevronMode: 0     // 0 off, 1 flow toward lead (yellow), 2 toward ego (red)
    // Distance tag over the lead vehicle (scene metres)
    property bool tagVisible: false
    property real tagX: 0
    property real tagZ: -20
    property real tagH: 1.6
    property string tagText: ""
    property string tagSub: ""
    property int tagLevel: 1
    // Red elements pulse at 2 Hz (spec UX-THR-04: 1–3 Hz).
    property real pulse: 0.0
    SequentialAnimation on pulse {
        running: root.clusterStyle
        loops: Animation.Infinite
        NumberAnimation { from: 0.0; to: 1.0; duration: 250; easing.type: Easing.InOutSine }
        NumberAnimation { from: 1.0; to: 0.0; duration: 250; easing.type: Easing.InOutSine }
    }
    // Chevron flow phase for the "closing fast" animation.
    property real flowPhase: 0.0
    NumberAnimation on flowPhase {
        running: root.clusterStyle && root.chevronMode !== 0
        from: 0.0; to: 1.0; duration: 1200; loops: Animation.Infinite
    }

    // Camera field of view (deg); the cluster uses a longer lens.
    property real fovDeg: 44
    // Optional soft-edge mask (an Item with alpha) applied to the 3D view only,
    // so the road dissolves into whatever is behind the scene.
    property Item fadeMask: null
    readonly property color vehicleBase: "#B9BDC9"

    // Neutral light-grey traffic, tinted toward the threat colour
    // (spec UX-3D-08 / §5: 18 % green, 38 % yellow, 60 % red).
    function vehicleColor(level) {
        if (level <= 0)
            return root.vehicleBase
        const c = threatColor(level)
        const a = level >= 3 ? 0.36 : (level === 2 ? 0.28 : 0.18)
        return Qt.tint(root.vehicleBase, Qt.rgba(c.r, c.g, c.b, a))
    }

    // Replace every material under a loaded GLB with one shared material so
    // traffic reads as clean neutral shapes the threat tint can colour.
    function _paintTree(node, mat) {
        let n = 0
        const kids = node.children
        if (!kids)
            return 0
        for (let i = 0; i < kids.length; i++) {
            const k = kids[i]
            if (k.objectName === "halo")
                continue
            if (k.materials !== undefined && k.materials.length !== undefined) {
                k.materials = [mat]
                n++
            }
            n += _paintTree(k, mat)
        }
        return n
    }
    function paintRig(rig, mat) {
        let n = 0
        const kids = rig.children
        for (let i = 0; i < kids.length; i++) {
            const k = kids[i]
            if (k.status !== undefined && k.status === RuntimeLoader.Success)
                n += _paintTree(k, mat)
        }
        return n > 0
    }

    function threatColor(level) {
        return level >= 3 ? root.threatRed : (level === 2 ? root.threatYellow
               : (level === 1 ? root.threatGreen : "#B9BDC9"))
    }
    function laneStateColor(state) {
        return state === "warning" ? root.threatRed
             : (state === "caution" ? root.threatYellow
             : (state === "off" ? root.offLine : root.neutralLine))
    }
    readonly property color cipoGlow: cipoStatus === "DANGER" ? "#e23a3c"
                                     : (cipoStatus === "WARNING" ? "#e09a20" : "#3ec8ff")
    readonly property color corridorColor: cipoStatus === "DANGER" ? Qt.rgba(0.95, 0.18, 0.32, 0.72)
                                     : (cipoStatus === "WARNING" ? Qt.rgba(1.0, 0.72, 0.08, 0.70)
                                                                 : Qt.rgba(0.24, 0.86, 0.52, 0.92))
    readonly property color laneColor: "#F4F7FB"
    readonly property color centerColor: "#D8FFE8"
    property string overlayHint: "Phase 2 — loading GLB"
    property url egoGltf: ""
    property url skodaGltf: ""
    property url teslaGltf: ""
    property url shcGltf: ""
    property url dodgeGltf: ""
    property real egoScale: 1.0
    property real skodaScale: 100.0
    property real teslaScale: 1.0
    property real teslaRotX: -90
    property real teslaRotY: 0
    property real teslaRotZ: 0
    property real teslaY: 0
    property bool teslaFitted: false
    property real shcScale: 0.01
    property real shcRotY: 90
    property real dodgeScale: 0.05
    property real egoRotX: 0
    property real egoRotY: 180
    property real egoRotZ: 0
    property real egoY: 0.3
    // Lane-anchored render: the lane is pinned and the ego moves within it.
    property real egoX: 0
    property real laneSlide: 0
    property real egoYawDeg: 0
    property real worldYawDeg: 0
    property bool laneValid: false
    property bool laneHeld: false
    property real targetCarLength: 4.6
    property real targetTruckLength: 10.0
    property bool egoFitted: false
    property bool skodaFitted: false
    property bool shcFitted: false
    property bool dodgeFitted: false
    property real skodaY: 0
    property real shcY: 0
    property real dodgeY: 0
    property string egoDebug: "bounds pending"
    property string trafficJson: "[]"
    property int trafficCount: 0
    property string corridorJson: "[]"
    property string laneJson: "[]"
    property string dashJson: "[]"
    property string centerJson: "[]"
    property string edgeJson: "[]"
    property string trailJson: "[]"

    onTrafficJsonChanged: applyTraffic()
    onTeslaYChanged: applyTraffic()
    onSkodaYChanged: applyTraffic()
    onShcYChanged: applyTraffic()
    onDodgeYChanged: applyTraffic()
    onCorridorJsonChanged: applySegPool(corrRep, root.corridorJson, 0.045)
    onLaneJsonChanged: applyLanes()
    onDashJsonChanged: applyDashes()
    onLaneLeftStateChanged: paintDashColors()
    onLaneRightStateChanged: paintDashColors()
    onCenterJsonChanged: applyCenter()
    onEdgeJsonChanged: applyEdges()
    onTrailJsonChanged: applyTrail()
    onChevronJsonChanged: applyChevrons()
    onShowLaneLinesChanged: { applyLanes(); applyDashes(); applyEdges(); applyTrail() }
    onCinematicRoadChanged: { applyLanes(); applyDashes(); applyEdges(); applyTrail() }
    onEnvModeChanged: refreshEnv()
    onClockHourChanged: refreshEnv()

    Component.onCompleted: {
        syncClockHour()
        refreshEnv()
        applySegPool(corrRep, root.corridorJson, 0.045)
        applyLanes()
        applyDashes()
        applyCenter()
        applyEdges()
        applyTrail()
        applyChevrons()
    }

    function applyChevrons() {
        let rows = []
        try { rows = JSON.parse(root.chevronJson || "[]") } catch (e) { rows = [] }
        const n = chevRep.count
        for (let i = 0; i < n; i++) {
            const m = chevRep.objectAt(i)
            if (!m)
                continue
            if (i < rows.length) {
                const r = rows[i]
                m.visible = true
                m.position = Qt.vector3d(r.x, 0.05, r.z)
                // Rectangle lies flat (rx -90); yaw follows the lane, flip
                // 180° so red chevrons point back at the ego.
                m.eulerRotation = Qt.vector3d(-90, r.yaw + (r.back ? 180 : 0), 0)
                m.scale = Qt.vector3d((r.w || 1.6) / 100.0, (r.len || 0.9) / 100.0, 1)
                m.seq = r.i | 0
            } else {
                m.visible = false
            }
        }
    }

    function syncClockHour() {
        const d = new Date()
        root.clockHour = d.getHours() + d.getMinutes() / 60.0 + d.getSeconds() / 3600.0
    }

    function cycleEnvMode() {
        const modes = ["auto", "day", "dusk", "night", "demo"]
        const i = modes.indexOf(root.envMode)
        root.envMode = modes[(i < 0 ? 0 : i + 1) % modes.length]
    }

    function _lerp(a, b, t) { return a + (b - a) * t }

    function _lerpColor(a, b, t) {
        return Qt.rgba(
            a.r + (b.r - a.r) * t,
            a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t,
            1.0
        )
    }

    function _palette(phase, demoBoost) {
        // Soft sky + mist + sun look; demoBoost amplifies saturation/brightness.
        if (phase === "night") {
            return {
                top: "#050810", mid: "#0b1220", bot: "#1a2740",
                clear: "#050810",
                light: "#c8d6e8", ambient: "#1c2430",
                bright: demoBoost ? 0.55 : 0.42,
                sun: "#d8e6ff", asphalt: "#0a0c10", shoulder: "#07090c",
                mist: "#1a2740", mistOp: 0.42, moon: true, glow: demoBoost ? 1.35 : 1.0,
                mtnFar: "#121820", mtnNear: "#080c10"
            }
        }
        if (phase === "dusk") {
            return {
                top: "#1a1038", mid: "#b04838", bot: "#f0a060",
                clear: "#1a1028",
                light: "#ffd0a8", ambient: "#5a4050",
                bright: demoBoost ? 1.55 : 1.15,
                sun: "#ffb060", asphalt: "#121018", shoulder: "#0e0c12",
                mist: "#e09060", mistOp: demoBoost ? 0.55 : 0.40, moon: false,
                glow: demoBoost ? 1.8 : 1.25,
                mtnFar: "#2a1830", mtnNear: "#120c18"
            }
        }
        // day — stronger top→horizon contrast so the sky does not read as flat fill
        return {
            top: "#1e5aaa", mid: "#5a9ed8", bot: "#c5e2f8",
            clear: "#5a90c0",
            light: "#fff8ee", ambient: "#6a7a88",
            bright: demoBoost ? 1.75 : 1.40,
            sun: "#ffe8a0", asphalt: "#12161c", shoulder: "#101812",
            mist: "#b8d4ea", mistOp: demoBoost ? 0.45 : 0.32, moon: false,
            glow: demoBoost ? 1.55 : 1.1,
            mtnFar: "#3d5568", mtnNear: "#1c2832"
        }
    }

    function _phaseFromHour(h) {
        // Night 19–6, dusk 15–19, day otherwise (morning shares day palette).
        if (h >= 19.0 || h < 6.0)
            return "night"
        if (h >= 15.0)
            return "dusk"
        return "day"
    }

    function _asCol(c) {
        return (typeof c === "object") ? c : Qt.color(c)
    }

    function _blendPalettes(a, b, t) {
        return {
            top: _lerpColor(_asCol(a.top), _asCol(b.top), t),
            mid: _lerpColor(_asCol(a.mid), _asCol(b.mid), t),
            bot: _lerpColor(_asCol(a.bot), _asCol(b.bot), t),
            clear: _lerpColor(_asCol(a.clear), _asCol(b.clear), t),
            light: _lerpColor(_asCol(a.light), _asCol(b.light), t),
            ambient: _lerpColor(_asCol(a.ambient), _asCol(b.ambient), t),
            bright: _lerp(a.bright, b.bright, t),
            sun: _lerpColor(_asCol(a.sun), _asCol(b.sun), t),
            asphalt: _lerpColor(_asCol(a.asphalt), _asCol(b.asphalt), t),
            shoulder: _lerpColor(_asCol(a.shoulder), _asCol(b.shoulder), t),
            mist: _lerpColor(_asCol(a.mist), _asCol(b.mist), t),
            mistOp: _lerp(a.mistOp, b.mistOp, t),
            moon: t > 0.5 ? b.moon : a.moon,
            glow: _lerp(a.glow, b.glow, t),
            mtnFar: _lerpColor(_asCol(a.mtnFar), _asCol(b.mtnFar), t),
            mtnNear: _lerpColor(_asCol(a.mtnNear), _asCol(b.mtnNear), t)
        }
    }

    function _autoPalette(h) {
        // Smooth crossfades near phase boundaries so Auto does not hard-cut.
        if (h >= 5.0 && h < 7.0)
            return _blendPalettes(_palette("night", false), _palette("day", false), (h - 5.0) / 2.0)
        if (h >= 14.5 && h < 16.5)
            return _blendPalettes(_palette("day", false), _palette("dusk", false), (h - 14.5) / 2.0)
        if (h >= 18.0 && h < 20.0)
            return _blendPalettes(_palette("dusk", false), _palette("night", false), (h - 18.0) / 2.0)
        return _palette(_phaseFromHour(h), false)
    }

    function _applySunOrbit(h, pal) {
        // Day arc 06–18; night moon arc 18–06 (wrapped).
        let elev = 0.35
        let azim = 0.0
        if (!pal.moon) {
            const p = Math.max(0.0, Math.min(1.0, (h - 6.0) / 12.0))
            elev = Math.sin(Math.PI * p)
            azim = _lerp(-58.0, 58.0, p)
        } else {
            let hn = h
            if (hn < 6.0)
                hn += 24.0
            const p = Math.max(0.0, Math.min(1.0, (hn - 18.0) / 12.0))
            elev = 0.28 + 0.35 * Math.sin(Math.PI * p)
            azim = _lerp(50.0, -50.0, p)
        }
        root.sunElev = elev
        root.sunAzim = azim
        root.sunIsMoon = !!pal.moon
        root.sunGlowScale = pal.glow
        root.sunColor = (typeof pal.sun === "object") ? pal.sun : Qt.color(pal.sun)

        const dist = 98.0
        const elevDeg = 6.0 + elev * (pal.moon ? 32.0 : 48.0)
        const az = azim * Math.PI / 180.0
        const el = elevDeg * Math.PI / 180.0
        root.sunPosX = Math.sin(az) * dist * Math.cos(el)
        root.sunPosY = Math.sin(el) * dist
        root.sunPosZ = -Math.cos(az) * dist * Math.cos(el)
    }

    function refreshEnv() {
        const mode = root.envMode
        const demo = (mode === "demo")
        let phase = "day"
        let pal
        if (mode === "auto") {
            phase = _phaseFromHour(root.clockHour)
            pal = _autoPalette(root.clockHour)
            root.envLabel = "Env Auto"
        } else if (mode === "demo") {
            // Demo locks a rich golden-hour look for recordings.
            phase = "dusk"
            pal = _palette("dusk", true)
            root.envLabel = "Env Demo"
        } else {
            phase = mode
            pal = _palette(mode, false)
            root.envLabel = "Env " + mode.charAt(0).toUpperCase() + mode.slice(1)
        }
        root.envPhase = phase

        const asColor = _asCol
        root.skyTopColor = asColor(pal.top)
        root.skyMidColor = asColor(pal.mid)
        root.skyBotColor = asColor(pal.bot)
        root.clearCol = asColor(pal.clear)
        root.lightCol = asColor(pal.light)
        root.ambientCol = asColor(pal.ambient)
        root.lightBright = pal.bright
        root.asphaltCol = asColor(pal.asphalt)
        root.shoulderCol = asColor(pal.shoulder)
        root.mistCol = asColor(pal.mist)
        root.mistOpacity = pal.mistOp
        root.mountainFarCol = asColor(pal.mtnFar)
        root.mountainNearCol = asColor(pal.mtnNear)

        // Sun hour: Auto uses clock; fixed presets use a canonical hour.
        let h = root.clockHour
        if (mode === "day")
            h = 12.0
        else if (mode === "dusk" || mode === "demo")
            h = 17.2
        else if (mode === "night")
            h = 22.0
        _applySunOrbit(h, {
            moon: pal.moon,
            glow: pal.glow,
            sun: pal.sun
        })
    }

    Timer {
        interval: 30000
        running: true
        repeat: true
        onTriggered: {
            if (root.envMode === "auto")
                root.syncClockHour()
        }
    }

    // Screen-space sun/moon (azimuth → X, elevation → height above horizon band).
    readonly property real sunDiscSize: (root.sunIsMoon ? 22 : 28) * root.sunGlowScale
    readonly property real sunScreenX: width * (0.5 + root.sunAzim / 145.0) - sunDiscSize * 1.3
    // Keep the disc in the visible sky band (above the road horizon ~34% down).
    readonly property real sunScreenY: {
        const y = height * (0.33 - root.sunElev * 0.20) - sunDiscSize * 1.3
        return Math.max(6, Math.min(height * 0.36, y))
    }

    // Scenic automotive horizon. Kept lightweight: gradients and vector silhouettes.
    Rectangle {
        id: skyBackdrop
        anchors.fill: parent
        z: 0
        visible: !root.clusterStyle
        gradient: Gradient {
            GradientStop { position: 0.0; color: root.scenicView ? root.skyTopColor : "#070b12" }
            GradientStop { position: 0.42; color: root.scenicView ? root.skyMidColor : "#0c141e" }
            GradientStop { position: 1.0; color: root.scenicView ? root.skyBotColor : "#101820" }
        }
    }

    Item {
        id: sunGlow
        z: 1
        visible: root.scenicView && !root.clusterStyle
        x: root.sunScreenX
        y: root.sunScreenY
        width: root.sunDiscSize * 2.6
        height: width
        Rectangle {
            anchors.fill: parent
            radius: width / 2
            color: Qt.rgba(root.sunColor.r, root.sunColor.g, root.sunColor.b, 0.13)
        }
        Rectangle {
            anchors.centerIn: parent
            width: parent.width * 0.38
            height: width
            radius: width / 2
            color: root.sunColor
        }
    }

    // Distant mountain ranges along the horizon (replaces the fake mist slab).
    Item {
        id: mountainLayer
        anchors.left: parent.left
        anchors.right: parent.right
        z: 1
        y: parent.height * 0.26
        height: parent.height * 0.22
        opacity: 0.0
        visible: false

        // Far range — lighter, softer, sits behind near peaks
        Shape {
            anchors.fill: parent
            ShapePath {
                fillColor: root.mountainFarCol
                strokeWidth: 0
                startX: 0
                startY: mountainLayer.height
                PathLine { x: mountainLayer.width * 0.00; y: mountainLayer.height * 0.72 }
                PathLine { x: mountainLayer.width * 0.10; y: mountainLayer.height * 0.38 }
                PathLine { x: mountainLayer.width * 0.18; y: mountainLayer.height * 0.55 }
                PathLine { x: mountainLayer.width * 0.28; y: mountainLayer.height * 0.22 }
                PathLine { x: mountainLayer.width * 0.38; y: mountainLayer.height * 0.48 }
                PathLine { x: mountainLayer.width * 0.50; y: mountainLayer.height * 0.18 }
                PathLine { x: mountainLayer.width * 0.62; y: mountainLayer.height * 0.42 }
                PathLine { x: mountainLayer.width * 0.74; y: mountainLayer.height * 0.12 }
                PathLine { x: mountainLayer.width * 0.86; y: mountainLayer.height * 0.40 }
                PathLine { x: mountainLayer.width * 0.96; y: mountainLayer.height * 0.28 }
                PathLine { x: mountainLayer.width * 1.00; y: mountainLayer.height * 0.58 }
                PathLine { x: mountainLayer.width; y: mountainLayer.height }
                PathLine { x: 0; y: mountainLayer.height }
            }
        }

        // Near range — darker silhouette in front
        Shape {
            anchors.fill: parent
            ShapePath {
                fillColor: root.mountainNearCol
                strokeWidth: 0
                startX: 0
                startY: mountainLayer.height
                PathLine { x: mountainLayer.width * 0.00; y: mountainLayer.height * 0.88 }
                PathLine { x: mountainLayer.width * 0.08; y: mountainLayer.height * 0.58 }
                PathLine { x: mountainLayer.width * 0.16; y: mountainLayer.height * 0.70 }
                PathLine { x: mountainLayer.width * 0.26; y: mountainLayer.height * 0.45 }
                PathLine { x: mountainLayer.width * 0.36; y: mountainLayer.height * 0.68 }
                PathLine { x: mountainLayer.width * 0.48; y: mountainLayer.height * 0.52 }
                PathLine { x: mountainLayer.width * 0.58; y: mountainLayer.height * 0.72 }
                PathLine { x: mountainLayer.width * 0.70; y: mountainLayer.height * 0.40 }
                PathLine { x: mountainLayer.width * 0.82; y: mountainLayer.height * 0.62 }
                PathLine { x: mountainLayer.width * 0.92; y: mountainLayer.height * 0.50 }
                PathLine { x: mountainLayer.width * 1.00; y: mountainLayer.height * 0.78 }
                PathLine { x: mountainLayer.width; y: mountainLayer.height }
                PathLine { x: 0; y: mountainLayer.height }
            }
        }

        // Soft atmospheric wash over the foothills (not a hard slab)
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: parent.height * 0.55
            gradient: Gradient {
                GradientStop {
                    position: 0.0
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b, 0.0)
                }
                GradientStop {
                    position: 0.55
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b,
                                   root.mistOpacity * 0.35)
                }
                GradientStop {
                    position: 1.0
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b,
                                   root.mistOpacity * 0.55)
                }
            }
        }
    }

    function applyLanes() {
        if (!root.showLaneLines) {
            applySegPool(laneRep, "[]", 0.05)
            return
        }
        if (!root.cinematicRoad) {
            applySegPool(laneRep, root.laneJson, 0.05)
            return
        }
        // Film: ego-adjacent (pal 0) is drawn as white dashes instead.
        let rows = []
        try { rows = JSON.parse(root.laneJson || "[]") } catch (e) { rows = [] }
        rows = rows.filter(function (r) { return (r.c | 0) !== 0 })
        applySegPool(laneRep, JSON.stringify(rows), 0.05)
    }

    function dashColor(pal) {
        if (!root.clusterStyle)
            return root.laneColor
        const p = pal | 0
        if (p === 1)
            return root.laneStateColor(root.laneLeftState)
        if (p === 2)
            return root.laneStateColor(root.laneRightState)
        return root.neutralLine
    }

    // Repeater3D materials do not see the delegate's `pal` or the lane-state
    // properties, so a binding stays on the white neutral colour. Paint the
    // departing line from here whenever the dashes or the warning change.
    function paintDashColors() {
        const n = dashRep.count
        for (let i = 0; i < n; i++) {
            const m = dashRep.objectAt(i)
            if (!m || !m.materials || m.materials.length < 1)
                continue
            m.materials[0].baseColor = root.dashColor(m.pal | 0)
        }
    }

    function applyDashes() {
        const src = (root.cinematicRoad && root.dashJson) ? root.dashJson : "[]"
        applySegPool(dashRep, src, 0.03)
        paintDashColors()
    }

    function applyCenter() {
        const src = root.centerJson ? root.centerJson : "[]"
        applySegPool(centerRep, src, 0.07)
    }

    function applyEdges() {
        // Road edges come only from stable, hysteretic lane slots. There is no
        // static ±5.35 fallback: it used to pop in on every detection miss and
        // cross the real edge on curves.
        applySegPool(edgeRep, root.cinematicRoad ? root.edgeJson : "[]", 0.028)
    }

    function applyTrail() {
        applySegPool(trailRep, "[]", 0.02)
    }

    function applyTraffic() {
        let rows = []
        try { rows = JSON.parse(root.trafficJson || "[]") } catch (e) { rows = [] }
        root.debugRows = rows
        const pools = [
            [skoda0, skoda1, skoda2, skoda3, skoda4, skoda5, skoda6, skoda7],
            [tesla0],
            [shc0],
            [truck0, truck1]
        ]
        const occupied = [
            [false, false, false, false, false, false, false, false],
            [false],
            [false],
            [false, false]
        ]
        for (let i = 0; i < rows.length; i++) {
            const r = rows[i]
            let kind = r.kind | 0
            let slot = r.slot | 0
            if (kind < 0 || kind >= pools.length)
                continue
            if (slot < 0 || slot >= pools[kind].length)
                continue
            const node = pools[kind][slot]
            occupied[kind][slot] = true
            const py = (kind === 1) ? root.teslaY
                     : (kind === 0) ? root.skodaY
                     : (kind === 2) ? root.shcY
                     : root.dodgeY
            // Position before showing, so a pooled mesh never draws at the
            // ego origin on the frame it is assigned.
            node.x = r.posX
            node.y = py
            node.z = r.posZ
            node.visible = true
            node.eulerRotation = Qt.vector3d(0, (r.yawDeg !== undefined ? r.yawDeg : root.egoRotY), 0)
            node.threat = (r.thr | 0)
            node.primary = !!r.prim
            node.carLen = (kind === 3) ? root.targetTruckLength : root.targetCarLength
            // Far cars otherwise shrink to specks in the chase camera.
            const z = Math.max(0.0, -r.posZ)
            const boost = (z > 18.0) ? Math.min(1.9, 1.0 + 0.016 * (z - 18.0)) : 1.0
            node.scale = Qt.vector3d(boost, boost, boost)
        }
        for (let k = 0; k < pools.length; k++) {
            for (let j = 0; j < pools[k].length; j++) {
                if (!occupied[k][j]) {
                    pools[k][j].visible = false
                    pools[k][j].threat = 0
                    pools[k][j].scale = Qt.vector3d(1, 1, 1)
                }
            }
        }
        root.trafficCount = rows.length
    }

    function applySegPool(rep, jsonStr, yLift) {
        let rows = []
        try { rows = JSON.parse(jsonStr || "[]") } catch (e) { rows = [] }
        const n = rep.count
        for (let i = 0; i < n; i++) {
            const m = rep.objectAt(i)
            if (!m)
                continue
            if (i < rows.length) {
                const r = rows[i]
                m.visible = true
                m.position = Qt.vector3d(r.x, yLift, r.z)
                m.eulerRotation = Qt.vector3d(0, r.yaw, 0)
                m.scale = Qt.vector3d((r.w || 0.18) / 100.0, 0.00022, (r.len || 2.5) / 100.0)
                if (m.pal !== undefined)
                    m.pal = r.c | 0
            } else {
                m.visible = false
            }
        }
    }

    readonly property real lookPitch: -(pitchDeg + calibPitch)
    readonly property real camHeight: camH * (calibH / 1.6)
    readonly property real camZ: camDist / Math.max(zoomFactor, 0.15)
    readonly property real yawRad: yawDeg * Math.PI / 180.0

    function fitEgoFromBounds() {
        const mn = egoCar.bounds.minimum
        const mx = egoCar.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        root.egoDebug = "raw " + dx.toFixed(3) + "×" + dy.toFixed(3) + "×" + dz.toFixed(3)
        if (longest < 1e-4)
            return
        if (longest < 1.5) {
            const s = root.targetCarLength / longest
            root.egoScale = s
            root.egoY = -mn.y * s
            root.egoDebug += "  → scale " + s.toFixed(1)
            // Never copy Audi fit onto traffic — Skoda/SHC/Dodge have their own units.
        } else if (longest > 20.0) {
            const s = root.targetCarLength / longest
            root.egoScale = s
            root.egoY = -mn.y * s
            root.egoDebug += "  → shrink " + s.toFixed(3)
        } else {
            root.egoDebug += "  keep scale " + root.egoScale.toFixed(1)
        }
        root.egoFitted = true
        console.log("[BEV] ego fit", root.egoDebug)
    }

    function fitMeshToCarLength(loader, targetLen) {
        if (!loader || loader.status !== RuntimeLoader.Success)
            return null
        const mn = loader.bounds.minimum
        const mx = loader.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        if (longest < 1e-4)
            return null
        const want = (targetLen === undefined || targetLen === null) ? root.targetCarLength : targetLen
        const s = want / longest
        return { s: s, y: -Math.min(mn.y, mx.y) * s, longest: longest }
    }

    function fitSkodaFromBounds() {
        const f = fitMeshToCarLength(skodaLoader)
        if (!f)
            return
        root.skodaScale = f.s
        root.skodaY = f.y
        root.skodaFitted = true
        console.log("[BEV] skoda fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitShcFromBounds() {
        const f = fitMeshToCarLength(shcLoader)
        if (!f)
            return
        root.shcScale = f.s
        root.shcY = f.y
        root.shcFitted = true
        console.log("[BEV] shc fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitDodgeFromBounds() {
        const f = fitMeshToCarLength(dodgeLoader, root.targetTruckLength)
        if (!f)
            return
        root.dodgeScale = f.s
        root.dodgeY = f.y
        root.dodgeFitted = true
        console.log("[BEV] truck fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitTeslaFromBounds() {
        const mn = teslaCar.bounds.minimum
        const mx = teslaCar.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        if (longest < 1e-4)
            return
        const s = root.targetCarLength / longest
        root.teslaScale = s
        // teslaFix is Rx(-90): local (x,y,z) → (x, z, -y). World up is local Z.
        const worldYmin = Math.min(mn.z, mx.z)
        root.teslaY = -worldYmin * s
        root.teslaFitted = true
        console.log("[BEV] tesla fit raw", dx.toFixed(3), dy.toFixed(3), dz.toFixed(3),
                    "mn", mn.x.toFixed(2), mn.y.toFixed(2), mn.z.toFixed(2),
                    "→ scale", s.toFixed(3), "y", root.teslaY.toFixed(3))
    }

    View3D {
        id: view
        anchors.fill: parent
        z: 2
        camera: sceneCamera
        layer.enabled: root.fadeMask !== null
        layer.effect: MultiEffect {
            maskEnabled: true
            maskSource: root.fadeMask
            // threshold 0.5 + spread 1.0 = soft ramp over the mask's alpha
            maskThresholdMin: 0.5
            maskSpreadAtMin: 1.0
        }

        environment: SceneEnvironment {
            // Let the 2D sky/sun behind this view show through empty pixels.
            backgroundMode: SceneEnvironment.Transparent
            clearColor: "#00000000"
            antialiasingMode: SceneEnvironment.NoAA
        }

        PerspectiveCamera {
            id: sceneCamera
            // Orbit around ego: move the camera, don't just yaw in place.
            position: Qt.vector3d(
                root.panX + Math.sin(root.yawRad) * root.camZ,
                root.camHeight + root.panY,
                Math.cos(root.yawRad) * root.camZ
            )
            eulerRotation: Qt.vector3d(root.lookPitch, root.yawDeg, 0)
            fieldOfView: root.fovDeg
            clipNear: 0.3
            clipFar: 180
        }

        DirectionalLight {
            eulerRotation.x: -32
            eulerRotation.y: 22
            brightness: 0.92
            color: "#d4e2f0"
            castsShadow: false
            ambientColor: "#243040"
        }

        // Cluster: soft key light from behind the camera so the rear faces of
        // the ego and traffic (what the driver sees) read on the dark field.
        DirectionalLight {
            visible: root.clusterStyle
            eulerRotation.x: -28
            eulerRotation.y: 0
            brightness: 0.75
            color: "#E8ECFF"
            castsShadow: false
        }

        // 3rd-person cluster road: long vanishing-point strip, ego-lane only.
        Node {
            id: worldRig
            eulerRotation: Qt.vector3d(0, root.worldYawDeg, 0)

            // Dark void under the road so the sides fall off into the cluster.
            Model {
                visible: !root.clusterStyle
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(0.80, 1.20, 1)
                position: Qt.vector3d(0, -0.04, -40)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.shoulderCol
                    roughness: 1.0
                }
            }
            // Pavement going to the horizon (about 12 m wide × 90 m long).
            Model {
                visible: !root.clusterStyle
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(0.12, 0.90, 1)
                position: Qt.vector3d(0, 0, -40)
                materials: PrincipledMaterial {
                    baseColor: root.asphaltCol
                    roughness: 0.97
                    metalness: 0.0
                }
            }
            // Cluster road: three lanes, unlit, a shade above the background,
            // running from behind the camera to the horizon.
            Model {
                visible: root.clusterStyle
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(0.148, 1.40, 1)
                position: Qt.vector3d(0, 0, -50)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#171725"
                }
            }
            // Cluster: soft headlamp pool on the road ahead of the ego (≤ 35 % white).
            Model {
                visible: root.clusterStyle
                source: "#Rectangle"
                eulerRotation.x: -90
                position: Qt.vector3d(root.egoX, 0.012, -9.0)
                scale: Qt.vector3d(0.09, 0.16, 1)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColorMap: Texture { source: "assets/pool.png" }
                    baseColor: "#C8D0FF"
                    opacity: 0.20
                    alphaMode: PrincipledMaterial.Blend
                }
            }
            // Cluster HUD grid — sparse cyan (keep cheap on Orin).
            Repeater3D {
                model: root.clusterStyle ? 0 : 12
                Model {
                    source: "#Cube"
                    position: Qt.vector3d(0, 0.025, 6.0 - index * 6.5)
                    scale: Qt.vector3d(0.12, 0.00008, 0.0002)
                    materials: PrincipledMaterial {
                        lighting: PrincipledMaterial.NoLighting
                        baseColor: "#3ec8ff"
                        opacity: 0.34
                        alphaMode: PrincipledMaterial.Blend
                    }
                }
            }
            Repeater3D {
                model: root.clusterStyle ? 0 : 7
                Model {
                    source: "#Cube"
                    position: Qt.vector3d((index - 3) * 2.0, 0.025, -38)
                    scale: Qt.vector3d(0.00014, 0.00008, 0.90)
                    materials: PrincipledMaterial {
                        lighting: PrincipledMaterial.NoLighting
                        baseColor: "#3ec8ff"
                        opacity: 0.26
                        alphaMode: PrincipledMaterial.Blend
                    }
                }
            }

        Repeater3D {
            id: edgeRep
            model: 24
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#ebefff"
                }
            }
        }

        Repeater3D {
            id: dashRep
            model: 28
            Model {
                property int pal: 0
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    // Colour is applied in paintDashColors(). A binding here
                    // stays white because this material cannot read `pal`.
                    baseColor: root.neutralLine
                    opacity: (root.clusterStyle && ((pal === 1 && root.laneLeftState === "warning")
                              || (pal === 2 && root.laneRightState === "warning")))
                             ? 0.65 + 0.35 * root.pulse : 1.0
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }

        // Gap-band chevrons: yellow flow toward the lead (closing fast) or
        // red pointing back at the ego (collision warning). Spec §6.3/6.4.
        Repeater3D {
            id: chevRep
            model: 10
            Model {
                property int seq: 0
                visible: false
                source: "#Rectangle"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColorMap: Texture { source: "assets/chevron.png" }
                    baseColor: root.chevronMode === 2 ? root.threatRed : root.threatYellow
                    alphaMode: PrincipledMaterial.Blend
                    opacity: {
                        if (root.chevronMode === 2)
                            return 0.62 + 0.38 * root.pulse
                        let p = root.flowPhase - seq * 0.14
                        p = p - Math.floor(p)
                        return 0.12 + 0.78 * (p < 0.3 ? p / 0.3 : Math.max(0.0, 1.0 - (p - 0.3) / 0.7))
                    }
                }
            }
        }

        Repeater3D {
            id: trailRep
            model: 32
            Model {
                property int pal: 0
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: pal === 1 ? "#8aa4c8" : "#6e889e"
                    opacity: 0.55
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }

        // ---- metric debug overlay -------------------------------------
        // Ego centreline at x=0 plus 10 m ruler ticks, in scene metres. This
        // is the ground truth against which trafficJson positions are read:
        // if a car sits between the 30 and 40 m ticks its posZ really is -3x.
        Model {
            source: "#Cube"
            visible: root.debugMetric
            position: Qt.vector3d(0, 0.06, -0.5 * root.debugMaxZ)
            scale: Qt.vector3d(0.0004, 0.0002, root.debugMaxZ / 100.0)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: "#ff3b30"
                opacity: 0.75
                alphaMode: PrincipledMaterial.Blend
            }
        }
        Repeater3D {
            model: Math.floor(root.debugMaxZ / 10) + 1
            Model {
                source: "#Cube"
                visible: root.debugMetric && index > 0
                position: Qt.vector3d(0, 0.06, -index * 10.0)
                // 8 m wide tick, 12 cm deep: reads as a rung on the centreline.
                scale: Qt.vector3d(0.08, 0.0002, 0.0012)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#ff9500"
                    opacity: 0.65
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }

        Repeater3D {
            id: centerRep
            model: root.clusterStyle ? 0 : 18
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.centerColor
                }
            }
        }
        Repeater3D {
            id: corrRep
            model: 18
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.clusterStyle ? root.threatColor(root.gapLevel) : root.corridorColor
                    opacity: !root.clusterStyle ? 0.52
                             : (root.gapLevel >= 3 ? 0.14 + 0.06 * root.pulse
                             : (root.gapLevel === 2 ? 0.16 : 0.12))
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }
        Repeater3D {
            id: laneRep
            model: 36
            Model {
                property int pal: 0
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: pal === 1 ? "#b89440" : (pal === 2 ? "#7aa8b4" : "#78dcff")
                }
            }
        }

        Model {
            id: cipoRing
            visible: false
            source: "#Cylinder"
            position: Qt.vector3d(root.cipoX, 0.06, root.cipoZ)
            scale: Qt.vector3d(0.034, 0.0004, 0.034)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: root.cipoGlow
                opacity: 0.85
                alphaMode: PrincipledMaterial.Blend
            }
        }
        Model {
            id: cipoBeacon
            visible: false
            source: "#Cylinder"
            position: Qt.vector3d(root.cipoX, 1.35, root.cipoZ)
            scale: Qt.vector3d(0.0014, 0.026, 0.0014)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: root.cipoGlow
            }
        }
        Node {
            visible: false
            eulerRotation: Qt.vector3d(root.lookPitch, root.yawDeg, 0)
            Model {
                source: "#Rectangle"
                scale: Qt.vector3d(0.020, 0.0055, 1)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.cipoGlow
                }
            }
        }

        // Pooled traffic: one RuntimeLoader per slot, source set once.
        component TrafficRig: Node {
            id: rig
            visible: false
            property int threat: 0
            property bool primary: false
            property real carLen: 4.6
            property bool painted: false
            PrincipledMaterial {
                id: rigMat
                baseColor: root.vehicleColor(rig.threat)
                roughness: 0.38
                metalness: 0.05
                Behavior on baseColor { ColorAnimation { duration: 150 } }
            }
            Timer {
                interval: 250
                repeat: true
                running: root.clusterStyle && !rig.painted
                onTriggered: rig.painted = root.paintRig(rig, rigMat)
            }
            // No Behavior on x/z. A new mesh lives at (0, 0) until the first
            // detection, and an animation from there drives the GLB out of the
            // ego car. Python already eases noisy tracks; the mesh snaps to
            // the detected position.
            // Threat halo: soft footprint glow in the threat colour. Red pulses
            // at 2 Hz; the primary threat gets a wider, brighter halo.
            Model {
                objectName: "halo"
                visible: root.clusterStyle && rig.threat > 0
                source: "#Rectangle"
                eulerRotation.x: -90
                position: Qt.vector3d(0, 0.03, 0)
                scale: Qt.vector3d((rig.primary ? 3.3 : 2.9) / 100.0,
                                   (rig.carLen + (rig.primary ? 2.0 : 1.4)) / 100.0, 1)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColorMap: Texture { source: "assets/halo.png" }
                    baseColor: root.threatColor(rig.threat)
                    alphaMode: PrincipledMaterial.Blend
                    opacity: rig.threat >= 3 ? 0.22 + 0.10 * root.pulse
                           : (rig.threat === 2 ? 0.32 : 0.40)
                }
            }
        }
        TrafficRig {
            id: skoda0
            visible: false
            RuntimeLoader {
                id: skodaLoader
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda1
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda2
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda3
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda4
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda5
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda6
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda7
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: tesla0
            visible: false
            // Inner node: cancel Tesla GLB extra X-90. Outer node yaw = ego (180).
            Node {
                id: teslaFix
                eulerRotation: Qt.vector3d(root.teslaRotX, root.teslaRotY, root.teslaRotZ)
                RuntimeLoader {
                    id: teslaCar
                    source: root.teslaGltf
                    scale: Qt.vector3d(root.teslaScale, root.teslaScale, root.teslaScale)
                }
            }
        }

        Timer {
            id: teslaFitTimer
            interval: 33
            repeat: true
            running: teslaCar.status === RuntimeLoader.Success && !root.teslaFitted
            onTriggered: root.fitTeslaFromBounds()
        }
        Timer {
            id: skodaFitTimer
            interval: 33
            repeat: true
            running: skodaLoader.status === RuntimeLoader.Success && !root.skodaFitted
            onTriggered: root.fitSkodaFromBounds()
        }
        TrafficRig {
            id: shc0
            visible: false
            RuntimeLoader {
                id: shcLoader
                source: root.shcGltf
                scale: Qt.vector3d(root.shcScale, root.shcScale, root.shcScale)
                eulerRotation: Qt.vector3d(0, root.shcRotY, 0)
            }
        }
        Timer {
            id: shcFitTimer
            interval: 33
            repeat: true
            running: shcLoader.status === RuntimeLoader.Success && !root.shcFitted
            onTriggered: root.fitShcFromBounds()
        }
        TrafficRig {
            id: truck0
            visible: false
            RuntimeLoader {
                id: dodgeLoader
                source: root.dodgeGltf
                scale: Qt.vector3d(root.dodgeScale, root.dodgeScale, root.dodgeScale)
            }
        }
        TrafficRig {
            id: truck1
            visible: false
            RuntimeLoader {
                source: root.dodgeGltf
                scale: Qt.vector3d(root.dodgeScale, root.dodgeScale, root.dodgeScale)
            }
        }
        Timer {
            id: dodgeFitTimer
            interval: 33
            repeat: true
            running: dodgeLoader.status === RuntimeLoader.Success && !root.dodgeFitted
            onTriggered: root.fitDodgeFromBounds()
        }
        } // worldRig — live world flashes opposite the ego, then settles

        // Ego stays heading-up except for the brief turn yaw on this GLB.
        Model {
            visible: root.egoGltf.toString() === "" || egoCar.status === RuntimeLoader.Error
            source: "#Cube"
            position: Qt.vector3d(root.egoX, 0.65, 0)
            eulerRotation: Qt.vector3d(0, root.egoYawDeg, 0)
            scale: Qt.vector3d(0.018, 0.013, 0.043)
            materials: PrincipledMaterial {
                baseColor: "#00c8ff"
                roughness: 0.35
                metalness: 0.15
            }
        }
        RuntimeLoader {
            id: egoCar
            source: root.egoGltf
            visible: root.egoGltf.toString() !== "" && status !== RuntimeLoader.Error
            position: Qt.vector3d(root.egoX, root.egoY, 0)
            scale: Qt.vector3d(root.egoScale, root.egoScale, root.egoScale)
            eulerRotation: Qt.vector3d(root.egoRotX, root.egoRotY + root.egoYawDeg, root.egoRotZ)
        }
        Timer {
            id: egoFitTimer
            interval: 33
            repeat: true
            running: egoCar.status === RuntimeLoader.Success && !root.egoFitted
            onTriggered: root.fitEgoFromBounds()
        }
    }

    // ---- Distance tag over the lead vehicle (spec UX-GAP-02/03) --------
    property string uiFont: "Inter"
    readonly property vector3d tagPt: view.mapFrom3DScene(Qt.vector3d(root.tagX, root.tagH, root.tagZ))
    Rectangle {
        id: distTag
        z: 4
        visible: root.clusterStyle && root.tagVisible && root.tagPt.z > 0
        readonly property color edge: root.threatColor(root.tagLevel)
        readonly property bool big: root.tagLevel >= 3
        width: tagRow.implicitWidth + (big ? 30 : 24)
        height: big ? 40 : 32
        radius: height / 2
        x: root.tagPt.x - width / 2
        y: root.tagPt.y - height - 8
        color: Qt.rgba(0.027, 0.031, 0.043, 0.86)
        border.width: big ? 2 : 1.5
        border.color: edge
        Behavior on x { NumberAnimation { duration: 90 } }
        Behavior on y { NumberAnimation { duration: 90 } }

        Row {
            id: tagRow
            anchors.centerIn: parent
            spacing: 8
            Rectangle {
                visible: root.tagLevel >= 2
                width: distTag.big ? 20 : 16
                height: width
                radius: width / 2
                anchors.verticalCenter: parent.verticalCenter
                color: distTag.edge
                Text {
                    anchors.centerIn: parent
                    text: "!"
                    color: "#07080B"
                    font.family: root.uiFont
                    font.pixelSize: distTag.big ? 15 : 12
                    font.weight: Font.Black
                }
            }
            Text {
                text: root.tagText
                color: "#FFFFFF"
                font.family: root.uiFont
                font.pixelSize: distTag.big ? 19 : 16
                font.weight: Font.DemiBold
                font.features: { "tnum": 1 }
                anchors.verticalCenter: parent.verticalCenter
            }
            Text {
                visible: root.tagSub !== ""
                text: root.tagSub
                color: distTag.edge
                font.family: root.uiFont
                font.pixelSize: distTag.big ? 19 : 16
                font.weight: Font.DemiBold
                font.features: { "tnum": 1 }
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

    MouseArea {
        id: orbitArea
        enabled: root.interactive
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: toolbar.visible ? toolbar.top : parent.bottom
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        hoverEnabled: true
        property real lastX: 0
        property real lastY: 0

        onPressed: function (mouse) {
            lastX = mouse.x
            lastY = mouse.y
        }
        onPositionChanged: function (mouse) {
            const dx = mouse.x - lastX
            const dy = mouse.y - lastY
            lastX = mouse.x
            lastY = mouse.y
            if (mouse.buttons & Qt.LeftButton) {
                // Full 360° yaw (painter BEV used ±85° because fake perspective broke).
                root.yawDeg = root.yawDeg + dx * 0.35
                root.pitchDeg = Math.max(8, Math.min(85, root.pitchDeg - dy * 0.35))
            } else if (mouse.buttons & Qt.RightButton) {
                root.panX += dx * 0.03
                root.panY -= dy * 0.03
            }
        }
    }

    WheelHandler {
        enabled: root.interactive
        onWheel: function (event) {
            const f = event.angleDelta.y > 0 ? 1.12 : 0.88
            const z = root.zoomFactor * f
            if (z >= 0.4 && z <= 4.5)
                root.zoomFactor = z
            event.accepted = true
        }
    }

    // Per-object metric readout: prints each rendered car's (x, z) in metres
    // beside its model, so BEV placement can be checked against the
    // range_diagnostics.py CSV without instrumenting the renderer.
    Repeater {
        model: root.debugMetric ? root.debugRows : []
        delegate: Text {
            // mapFrom3DScene projects the metric scene point into viewport px,
            // so the label tracks the car under camera orbit and zoom.
            readonly property vector3d vp: view.mapFrom3DScene(
                Qt.vector3d(modelData.posX, 1.2, modelData.posZ))
            visible: vp.z > 0
            x: vp.x + 8
            y: vp.y - 8
            z: 4
            text: "#" + modelData.tid + "  x=" + modelData.posX.toFixed(2)
                  + "  z=" + (-modelData.posZ).toFixed(1) + " m"
            color: "#ffd60a"
            font.pixelSize: 10
            font.bold: true
            style: Text.Outline
            styleColor: "#000000"
        }
    }

    Column {
        visible: !root.clusterChrome
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 8
        spacing: 4
        z: 3

        Text {
            visible: root.debugMetric
            text: "DEBUG METRIC  centreline x=0, ticks 10 m"
            color: "#ff9500"
            font.pixelSize: 10
            font.bold: true
        }
        Text {
            text: "QT QUICK 3D BEV"
            color: "#c9d1d9"
            font.pixelSize: 11
            font.bold: true
        }
        Rectangle {
            width: cipoBadgeTxt.implicitWidth + 16
            height: 18
            radius: 3
            color: root.cipoStatus === "DANGER" ? "#da3633"
                 : (root.cipoStatus === "WARNING" ? "#d9822b"
                 : (root.cipoStatus === "DEGRADED" ? "#6e7681" : "#238636"))
            Text {
                id: cipoBadgeTxt
                anchors.centerIn: parent
                text: root.cipoVisible
                      ? ("CIPO: " + root.cipoStatus + "  " + root.cipoDist.toFixed(1) + "m")
                      : ("CIPO: " + root.cipoStatus)
                color: "#ffffff"
                font.pixelSize: 9
                font.bold: true
            }
        }
        Rectangle {
            width: ldwBadgeTxt.implicitWidth + 16
            height: 18
            radius: 3
            color: (root.ldwSide === "LEFT" || root.ldwSide === "RIGHT") ? "#d9822b" : "#21262d"
            Text {
                id: ldwBadgeTxt
                anchors.centerIn: parent
                text: (root.ldwSide === "LEFT" || root.ldwSide === "RIGHT")
                      ? ("LDW " + (root.ldwSide === "LEFT" ? "L" : "R"))
                      : "LDW"
                color: "#ffffff"
                font.pixelSize: 9
                font.bold: true
            }
        }
        Rectangle {
            width: fcwBadgeTxt.implicitWidth + 16
            height: 18
            radius: 3
            color: root.fcwLevel === "FCW+" ? "#da3633"
                 : (root.fcwLevel === "FCW" ? "#d9822b" : "#21262d")
            Text {
                id: fcwBadgeTxt
                anchors.centerIn: parent
                text: root.fcwLevel === "OFF" || root.fcwLevel === "" ? "FCW" : root.fcwLevel
                color: "#ffffff"
                font.pixelSize: 9
                font.bold: true
            }
        }
        Text {
            text: egoCar.status === RuntimeLoader.Error
                  ? ("GLB error: " + egoCar.errorString)
                  : (egoCar.status === RuntimeLoader.Success
                     ? (root.laneHeld ? "lane held — dead reckoning"
                        : (root.laneValid
                           ? "lane locked — ego centered"
                           : "no lane"))
                     : root.overlayHint)
            color: egoCar.status === RuntimeLoader.Error ? "#f85149"
                 : (root.laneHeld ? "#d9822b" : "#8b949e")
            font.pixelSize: 9
        }
    }

    Rectangle {
        id: toolbar
        visible: !root.clusterChrome
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 22
        color: "#b0161b22"
        z: 3

        Row {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 6
            spacing: 6

            Repeater {
                model: [
                    { key: "reset", label: "Reset" },
                    { key: "scenic", label: root.scenicView ? "Scene" : "Road" },
                    { key: "road", label: root.cinematicRoad ? "Film" : "Grid" },
                    { key: "lanes", label: root.showLaneLines ? "Lanes" : "No lanes" },
                    { key: "env", label: root.envLabel },
                    { key: "cal", label: "Cal" }
                ]
                Rectangle {
                    required property var modelData
                    width: chipText.implicitWidth + 10
                    height: 16
                    radius: 3
                    color: chipMouse.containsMouse ? "#30363d" : "#21262d"
                    Text {
                        id: chipText
                        anchors.centerIn: parent
                        text: parent.modelData.label
                        color: "#c9d1d9"
                        font.pixelSize: 9
                    }
                    MouseArea {
                        id: chipMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: {
                            const key = parent.modelData.key
                            if (key === "reset") {
                                root.pitchDeg = 8
                                root.yawDeg = 0
                                root.zoomFactor = 1.08
                                root.panX = 0
                                root.panY = 0
                                root.calibPitch = 0
                                root.calibH = 1
                            } else if (key === "scenic") {
                                root.scenicView = !root.scenicView
                            } else if (key === "road") {
                                root.cinematicRoad = !root.cinematicRoad
                            } else if (key === "env") {
                                root.cycleEnvMode()
                            } else if (key === "cal") {
                                root.showCalib = !root.showCalib
                            } else {
                                root.showLaneLines = !root.showLaneLines
                            }
                        }
                    }
                }
            }

            Text {
                visible: root.showCalib
                text: "P " + root.calibPitch.toFixed(0)
                color: "#8b949e"
                font.pixelSize: 9
            }
            Rectangle {
                visible: root.showCalib
                width: 70
                height: 5
                radius: 2
                color: "#2d3440"
                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: "#58a6ff"
                    y: -1.5
                    x: Math.max(0, Math.min(parent.width - 8,
                       (root.calibPitch + 10) / 20 * (parent.width - 8)))
                    MouseArea {
                        anchors.fill: parent
                        anchors.margins: -6
                        drag.target: parent
                        drag.axis: Drag.XAxis
                        drag.minimumX: 0
                        drag.maximumX: 62
                        onPositionChanged: {
                            parent.x = Math.max(0, Math.min(62, parent.x))
                            root.calibPitch = -10 + (parent.x / 62) * 20
                        }
                    }
                }
            }
            Text {
                visible: root.showCalib
                text: "H " + root.calibH.toFixed(1)
                color: "#8b949e"
                font.pixelSize: 9
            }
            Rectangle {
                visible: root.showCalib
                width: 54
                height: 5
                radius: 2
                color: "#2d3440"
                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: "#58a6ff"
                    y: -1.5
                    x: Math.max(0, Math.min(parent.width - 8,
                       (root.calibH - 1.0) / 2.0 * (parent.width - 8)))
                    MouseArea {
                        anchors.fill: parent
                        anchors.margins: -6
                        drag.target: parent
                        drag.axis: Drag.XAxis
                        drag.minimumX: 0
                        drag.maximumX: 46
                        onPositionChanged: {
                            parent.x = Math.max(0, Math.min(46, parent.x))
                            root.calibH = 1.0 + (parent.x / 46) * 2.0
                        }
                    }
                }
            }
        }
    }
}
