// ADAS instrument cluster — driver display.
//
// Layout and visual language follow the ADAS Cluster UI/UX spec: dark field
// inside a violet bezel, speed on the left, a live 3D driving scene in the
// centre, the front camera on the right, status bars top and bottom. Colour
// is spent only on state: green / yellow / red = threat (spec §5), cyan =
// information, orange = degraded or fault.
//
// Everything is authored on a fixed 1920 × 976 canvas and scaled uniformly to
// the window, so the cluster looks identical on a 1280 × 720 bench monitor and
// a 1920 × 720 cluster panel (letterboxed in the background colour).
//
// Data comes from the Python controller (src/ui/cluster_app.py) through the
// context property `cluster`: `cluster.hud` is one map refreshed per frame.

import QtQuick
import QtQuick.Shapes
import QtQuick.Dialogs

Item {
    id: app
    width: 1440
    height: 732

    readonly property var hud: cluster ? cluster.hud : ({})

    // ---------------------------------------------------------- tokens
    readonly property color bgBase: "#07080B"
    readonly property color bgSurface: "#12121C"
    readonly property color bezelA: "#6E64A8"
    readonly property color bezelB: "#9488E0"
    readonly property color violet: "#B66BFF"
    readonly property color textPrimary: "#FFFFFF"
    readonly property color textSecondary: "#9A9AB0"
    readonly property color textDisabled: "#4A4A5C"
    readonly property color infoCyan: "#4CD6FF"
    readonly property color threatGreen: "#4DFFB4"
    readonly property color threatYellow: "#FFD84D"
    readonly property color threatRed: "#FF5E7E"
    readonly property color systemOrange: "#FF9A4D"
    readonly property string uiFont: interRegular.status === FontLoader.Ready ? interRegular.font.family : "Sans"
    readonly property string displayFont: interDisplay.status === FontLoader.Ready ? interDisplay.font.family : uiFont

    FontLoader { id: interRegular; source: "fonts/Inter-Regular.ttf" }
    FontLoader { source: "fonts/Inter-Medium.ttf" }
    FontLoader { source: "fonts/Inter-SemiBold.ttf" }
    FontLoader { source: "fonts/Inter-Bold.ttf" }
    FontLoader { id: interDisplay; source: "fonts/InterDisplay-Bold.ttf" }
    FontLoader { source: "fonts/InterDisplay-SemiBold.ttf" }

    function levelColor(level) {
        return level >= 3 ? threatRed : (level === 2 ? threatYellow
               : (level === 1 ? threatGreen : textSecondary))
    }
    function stateColor(state) {
        // telltale / function state → colour
        if (state === "active") return threatRed
        if (state === "armed") return threatGreen
        if (state === "standby") return textSecondary
        if (state === "fault") return systemOrange
        return textDisabled
    }
    function val(key, fallback) {
        const v = hud ? hud[key] : undefined
        return (v === undefined || v === null) ? fallback : v
    }

    readonly property bool redActive: val("redActive", false)
    property real dimOpacity: redActive ? 0.4 : 1.0
    Behavior on dimOpacity { NumberAnimation { duration: 150 } }

    // 2 Hz pulse shared by every red element (spec UX-THR-04).
    property real pulse: 0
    SequentialAnimation on pulse {
        loops: Animation.Infinite
        NumberAnimation { from: 0; to: 1; duration: 250; easing.type: Easing.InOutSine }
        NumberAnimation { from: 1; to: 0; duration: 250; easing.type: Easing.InOutSine }
    }

    Rectangle { anchors.fill: parent; color: app.bgBase }

    // ============================================================ canvas
    Item {
        id: canvas
        width: 1920
        height: 976
        anchors.centerIn: parent
        scale: Math.min(app.width / width, app.height / height)
        transformOrigin: Item.Center

        // Ambient violet glow under the bezel.
        Shape {
            anchors.fill: parent
            ShapePath {
                strokeWidth: 0
                strokeColor: "transparent"
                fillGradient: RadialGradient {
                    centerX: 960; centerY: 880; centerRadius: 620
                    focalX: 960; focalY: 880
                    GradientStop { position: 0.0; color: Qt.rgba(110 / 255, 100 / 255, 168 / 255, 0.20) }
                    GradientStop { position: 1.0; color: Qt.rgba(110 / 255, 100 / 255, 168 / 255, 0.0) }
                }
                startX: 0; startY: 600
                PathLine { x: 1920; y: 600 }
                PathLine { x: 1920; y: 976 }
                PathLine { x: 0; y: 976 }
                PathLine { x: 0; y: 600 }
            }
        }

        // ------------------------------------------------------ bezel
        readonly property string bezelOpen: "M236 196 L78 470 Q62 500 82 532 L268 772 Q300 808 352 810 L960 832 L1568 810 Q1620 808 1652 772 L1838 532 Q1858 500 1842 470 L1684 196"
        Shape {
            id: bezelFill
            anchors.fill: parent
            ShapePath {
                strokeWidth: 0
                strokeColor: "transparent"
                fillGradient: LinearGradient {
                    x1: 0; y1: 150; x2: 0; y2: 832
                    GradientStop { position: 0.0; color: Qt.rgba(14 / 255, 13 / 255, 24 / 255, 0.0) }
                    GradientStop { position: 0.30; color: "#0E0D18" }
                    GradientStop { position: 1.0; color: "#121121" }
                }
                PathSvg { path: canvas.bezelOpen + " Q960 150 236 196 Z" }
            }
        }
        // Glow = three stacked strokes (no blur pass: cheap on Orin).
        Repeater {
            model: [ { w: 42, o: 0.04 }, { w: 32, o: 0.06 }, { w: 23, o: 0.09 },
                     { w: 15, o: 0.15 }, { w: 7, o: 1.0 } ]
            Shape {
                anchors.fill: parent
                opacity: modelData.o
                ShapePath {
                    fillColor: "transparent"
                    strokeWidth: modelData.w
                    joinStyle: ShapePath.RoundJoin
                    capStyle: ShapePath.RoundCap
                    strokeColor: app.bezelB
                    PathSvg { path: canvas.bezelOpen }
                }
            }
        }
        // Fade the bezel's open top ends into the background.
        Rectangle {
            x: 0; y: 150; width: 1920; height: 120
            gradient: Gradient {
                GradientStop { position: 0.0; color: app.bgBase }
                GradientStop { position: 1.0; color: Qt.rgba(7 / 255, 8 / 255, 11 / 255, 0.0) }
            }
        }

        // Red forward-collision vignette (spec UX-VIS-10).
        Item {
            anchors.fill: parent
            visible: app.val("vignette", false)
            opacity: 0.55 + 0.45 * app.pulse
            // Opaque strokes + item opacity: no seams where segments overlap.
            Repeater {
                model: [ { w: 64, o: 0.06 }, { w: 48, o: 0.09 }, { w: 34, o: 0.13 },
                         { w: 22, o: 0.20 }, { w: 11, o: 0.55 } ]
                Shape {
                    anchors.fill: parent
                    opacity: modelData.o
                    ShapePath {
                        fillColor: "transparent"
                        strokeColor: app.threatRed
                        strokeWidth: modelData.w
                        joinStyle: ShapePath.RoundJoin
                        capStyle: ShapePath.RoundCap
                        PathSvg { path: canvas.bezelOpen }
                    }
                }
            }
            Shape {
                anchors.fill: parent
                ShapePath {
                    strokeWidth: 0
                    strokeColor: "transparent"
                    fillGradient: RadialGradient {
                        centerX: 960; centerY: 520; centerRadius: 980
                        focalX: 960; focalY: 520
                        GradientStop { position: 0.55; color: Qt.rgba(1.0, 0.37, 0.49, 0.0) }
                        GradientStop { position: 0.80; color: Qt.rgba(1.0, 0.37, 0.49, 0.10) }
                        GradientStop { position: 1.0; color: Qt.rgba(1.0, 0.37, 0.49, 0.0) }
                    }
                    startX: 0; startY: 0
                    PathLine { x: 1920; y: 0 }
                    PathLine { x: 1920; y: 976 }
                    PathLine { x: 0; y: 976 }
                    PathLine { x: 0; y: 0 }
                }
            }
        }

        // ------------------------------------------------ 3D driving scene
        Item {
            id: sceneZone
            x: 430; y: 214; width: 1060; height: 520

            BevScene {
                id: bevScene
                objectName: "bevScene"
                anchors.fill: parent
                clusterStyle: true
                clusterChrome: true
                interactive: false
                uiFont: app.uiFont
                fadeMask: sceneMask
                visible: !app.val("showCamera", false)
            }
            // Soft-edged alpha mask: the road dissolves into the bezel
            // instead of ending at a rectangle.
            Image {
                id: sceneMask
                anchors.fill: parent
                source: "assets/scene_mask.png"
                visible: false
                layer.enabled: true
                smooth: true
            }

            // Full camera view (press C or tap the camera tile).
            Rectangle {
                anchors.fill: parent
                anchors.margins: 10
                visible: app.val("showCamera", false)
                radius: 18
                color: "#0B0B12"
                clip: true
                Image {
                    id: mainCamera
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    cache: false
                    asynchronous: false
                    source: parent.visible && app.val("hasFrame", false)
                            ? ("image://camera/" + app.val("frameId", 0)) : ""
                }
            }
        }

        // ------------------------------------------- speed-limit plate
        Item {
            id: limitPlate
            x: 556; y: 336
            width: 78; height: 102
            readonly property var limit: app.val("limit", null)
            readonly property bool kmh: app.val("units", "mph") === "kmh"
            readonly property int over: app.val("overLimit", 0)
            visible: limit !== null && !app.val("showCamera", false)
            opacity: app.val("limitConfirmed", true) ? 1.0 : 0.55

            // Outline ring: yellow over the limit, red (pulsing) well over it.
            Rectangle {
                anchors.fill: usPlate.visible ? usPlate : roundPlate
                anchors.margins: -7
                radius: usPlate.visible ? 12 : width / 2
                color: "transparent"
                border.width: 4
                border.color: limitPlate.over >= 3 ? app.threatRed : app.threatYellow
                visible: limitPlate.over >= 2
                opacity: limitPlate.over >= 3 ? 0.55 + 0.45 * app.pulse : 1.0
            }
            Rectangle {
                id: usPlate
                visible: !limitPlate.kmh
                anchors.fill: parent
                radius: 7
                color: "#F7F7F7"
                border.color: "#111111"
                border.width: 3
                Column {
                    anchors.centerIn: parent
                    spacing: 0
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: "SPEED"; color: "#111111"
                        font.family: app.uiFont; font.pixelSize: 12; font.weight: Font.Bold
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: "LIMIT"; color: "#111111"
                        font.family: app.uiFont; font.pixelSize: 12; font.weight: Font.Bold
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: limitPlate.limit === null ? "" : String(limitPlate.limit)
                        color: "#111111"
                        font.family: app.displayFont; font.pixelSize: 38; font.weight: Font.Black
                        font.features: { "tnum": 1 }
                    }
                }
            }
            Rectangle {
                id: roundPlate
                visible: limitPlate.kmh
                width: 78; height: 78
                radius: 39
                color: "#FFFFFF"
                border.color: "#E8344E"
                border.width: 8
                Text {
                    anchors.centerIn: parent
                    text: limitPlate.limit === null ? "" : String(limitPlate.limit)
                    color: "#07080B"
                    font.family: app.displayFont; font.pixelSize: 30; font.weight: Font.Black
                    font.features: { "tnum": 1 }
                }
            }
        }

        // ------------------------------------------------- alert banner
        Item {
            id: banner
            readonly property string kind: app.val("banner", "")
            readonly property string side: app.val("bannerSide", "")
            readonly property bool solid: kind === "fcw_plus"
            visible: kind !== ""
            // Takes the status pill's place: the pill is non-driving content
            // and hides while a red alert holds the banner (spec §7).
            x: 470; y: 172; width: 980; height: 72
            z: 10

            Rectangle {
                id: bannerPill
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                height: banner.solid ? 70 : 58
                width: bannerRow.implicitWidth + (banner.solid ? 72 : 52)
                radius: banner.solid ? 18 : 16
                color: banner.solid ? app.threatRed : Qt.rgba(26 / 255, 10 / 255, 18 / 255, 0.92)
                border.width: banner.solid ? 0 : 2
                border.color: app.threatRed
                scale: banner.visible ? 1.0 : 0.9
                Behavior on scale { NumberAnimation { duration: 120; easing.type: Easing.OutBack } }

                Row {
                    id: bannerRow
                    anchors.centerIn: parent
                    spacing: 16
                    // Collision glyph (car + impact marks)
                    Shape {
                        visible: banner.kind === "fcw" || banner.kind === "fcw_plus"
                        width: 48; height: 36
                        anchors.verticalCenter: parent.verticalCenter
                        scale: 1.05
                        ShapePath {
                            strokeWidth: 0
                            fillColor: banner.solid ? app.bgBase : app.threatRed
                            PathSvg { path: "M3 24 V18 L8 11 Q9.5 9 12 9 H22 Q24.5 9 26 11 L30 17 H32 Q35 17 35 20 V24 Z M6.5 26 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 Z M24.5 26 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 Z" }
                        }
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: banner.solid ? app.bgBase : app.threatRed
                            strokeWidth: 2.6
                            capStyle: ShapePath.RoundCap
                            PathSvg { path: "M38 9 L42 5 M39 15 H44 M38 21 L42 25" }
                        }
                    }
                    // Lane departure glyph (mirrored for the right side)
                    Shape {
                        visible: banner.kind === "ldw"
                        width: 36; height: 36
                        anchors.verticalCenter: parent.verticalCenter
                        transform: Scale { origin.x: 18; xScale: banner.side === "RIGHT" ? -1 : 1 }
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: app.threatRed; strokeWidth: 3; capStyle: ShapePath.RoundCap
                            PathSvg { path: "M5 3 V33" }
                        }
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: "#FFFFFF"; strokeWidth: 3; capStyle: ShapePath.RoundCap
                            PathSvg { path: "M31 3 V9 M31 15 V21 M31 27 V33" }
                        }
                        ShapePath {
                            strokeWidth: 0; fillColor: "#FFFFFF"
                            PathSvg { path: "M17 9 H21 Q25 9 25 13 V24 Q25 28 21 28 H17 Q13 28 13 24 V13 Q13 9 17 9 Z" }
                        }
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: app.threatRed; strokeWidth: 2.4
                            capStyle: ShapePath.RoundCap; joinStyle: ShapePath.RoundJoin
                            PathSvg { path: "M11 18 H8 M10 15 L7 18 L10 21" }
                        }
                    }
                    // Occupied adjacent lane glyph
                    Shape {
                        visible: banner.kind === "lane_change"
                        width: 40; height: 36
                        anchors.verticalCenter: parent.verticalCenter
                        transform: Scale { origin.x: 20; xScale: banner.side === "LEFT" ? -1 : 1 }
                        ShapePath {
                            strokeWidth: 0; fillColor: "#FFFFFF"
                            PathSvg { path: "M10 8 H15 Q19 8 19 12 V25 Q19 29 15 29 H10 Q6 29 6 25 V12 Q6 8 10 8 Z" }
                        }
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: app.threatRed; strokeWidth: 3; capStyle: ShapePath.RoundCap
                            PathSvg { path: "M25 10 Q30 18 25 26 M30 6 Q38 18 30 30" }
                        }
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: banner.kind === "fcw_plus" ? "BRAKE"
                            : (banner.kind === "fcw" ? "Collision warning"
                            : (banner.kind === "ldw" ? "Lane departure"
                            : (banner.kind === "lane_change" ? "Vehicle alongside" : "")))
                        color: banner.solid ? app.bgBase : "#FFFFFF"
                        font.family: banner.solid ? app.displayFont : app.uiFont
                        font.pixelSize: banner.solid ? 42 : 30
                        font.weight: banner.solid ? Font.Black : Font.DemiBold
                        font.letterSpacing: banner.solid ? 4 : 0
                    }
                }
            }
        }

        // ----------------------- non-driving content (dims on red alerts)
        Item {
            anchors.fill: parent
            opacity: app.dimOpacity

            // Source (top left)
            Row {
                x: 330; y: 190
                spacing: 14
                Rectangle {
                    width: 30; height: 30; radius: 15
                    anchors.verticalCenter: parent.verticalCenter
                    color: app.val("paused", false) ? app.textSecondary
                         : (app.val("sourceKind", "") === "sim" ? app.violet : "#FFFFFF")
                    Shape {
                        anchors.fill: parent
                        visible: !app.val("paused", false)
                        ShapePath {
                            strokeWidth: 0; fillColor: app.bgBase
                            PathSvg { path: "M12 9 L21 15 L12 21 Z" }
                        }
                    }
                    Row {
                        anchors.centerIn: parent
                        spacing: 4
                        visible: app.val("paused", false)
                        Rectangle { width: 4; height: 12; color: app.bgBase }
                        Rectangle { width: 4; height: 12; color: app.bgBase }
                    }
                }
                Item {
                    width: 300; height: 34
                    clip: true
                    Text {
                        id: sourceText
                        anchors.verticalCenter: parent.verticalCenter
                        text: app.val("sourceLabel", "")
                        color: "#FFFFFF"
                        font.family: app.uiFont; font.pixelSize: 24; font.weight: Font.Medium
                    }
                    Rectangle {
                        visible: sourceText.implicitWidth > parent.width
                        anchors.right: parent.right
                        width: 70; height: parent.height
                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: Qt.rgba(7 / 255, 8 / 255, 11 / 255, 0) }
                            GradientStop { position: 1.0; color: app.bgBase }
                        }
                    }
                }
            }

            // Perception status pill (top centre, reference ETA pill shape)
            Item {
                x: 740; y: 158; width: 440; height: 98
                opacity: banner.visible ? 0.0 : 1.0
                Behavior on opacity { NumberAnimation { duration: 120 } }
                Shape {
                    anchors.fill: parent
                    ShapePath {
                        strokeWidth: 0
                        strokeColor: "transparent"
                        fillGradient: LinearGradient {
                            x1: 0; y1: 0; x2: 0; y2: 98
                            GradientStop { position: 0.0; color: Qt.rgba(23 / 255, 22 / 255, 36 / 255, 0.2) }
                            GradientStop { position: 1.0; color: "#191828" }
                        }
                        PathSvg { path: "M0 0 H440 V46 Q440 98 388 98 H52 Q0 98 0 46 Z" }
                    }
                }
                Column {
                    anchors.centerIn: parent
                    anchors.verticalCenterOffset: 4
                    spacing: 6
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: app.val("statusCaption", "Perception")
                        color: app.textSecondary
                        font.family: app.uiFont; font.pixelSize: 15
                    }
                    Row {
                        anchors.horizontalCenter: parent.horizontalCenter
                        spacing: 12
                        Rectangle {
                            width: 10; height: 10; radius: 5
                            anchors.verticalCenter: parent.verticalCenter
                            color: app.val("laneTrack", "off") === "locked" ? app.threatGreen
                                 : (app.val("laneTrack", "off") === "held" ? app.threatYellow
                                 : app.systemOrange)
                        }
                        Text {
                            text: app.val("laneTrackText", "Lanes —")
                            color: "#FFFFFF"
                            font.family: app.uiFont; font.pixelSize: 24; font.weight: Font.DemiBold
                        }
                        Text {
                            text: "·"
                            color: app.textSecondary
                            font.family: app.uiFont; font.pixelSize: 24
                        }
                        Text {
                            text: app.val("objectsText", "0 tracked")
                            color: "#FFFFFF"
                            font.family: app.uiFont; font.pixelSize: 24; font.weight: Font.DemiBold
                            font.features: { "tnum": 1 }
                        }
                    }
                }
            }

            // SoC temperature + clock (top right)
            Row {
                x: 1356; y: 184
                spacing: 24
                Row {
                    spacing: 10
                    anchors.verticalCenter: parent.verticalCenter
                    visible: app.val("socTemp", null) !== null
                    Shape {
                        width: 26; height: 26
                        anchors.verticalCenter: parent.verticalCenter
                        ShapePath {
                            fillColor: "transparent"
                            strokeColor: app.val("socHot", false) ? app.systemOrange : "#FFFFFF"
                            strokeWidth: 2.2
                            PathSvg { path: "M7 5 H19 Q21 5 21 7 V19 Q21 21 19 21 H7 Q5 21 5 19 V7 Q5 5 7 5 Z M9 1 V5 M13 1 V5 M17 1 V5 M9 21 V25 M13 21 V25 M17 21 V25 M1 9 H5 M1 13 H5 M1 17 H5 M21 9 H25 M21 13 H25 M21 17 H25" }
                        }
                    }
                    Text {
                        id: socVal
                        text: app.val("socTemp", 0) + "°"
                        color: app.val("socHot", false) ? app.systemOrange : "#FFFFFF"
                        font.family: app.uiFont; font.pixelSize: 30; font.weight: Font.DemiBold
                        font.features: { "tnum": 1 }
                    }
                    Text {
                        text: "SoC"
                        color: app.textSecondary
                        anchors.baseline: socVal.baseline
                        font.family: app.uiFont; font.pixelSize: 17
                    }
                }
                Row {
                    spacing: 6
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        id: clockText
                        text: app.val("clock", "--:--")
                        color: "#FFFFFF"
                        font.family: app.uiFont; font.pixelSize: 36; font.weight: Font.DemiBold
                        font.features: { "tnum": 1 }
                    }
                    Text {
                        text: app.val("ampm", "")
                        color: app.textSecondary
                        anchors.baseline: clockText.baseline
                        font.family: app.uiFont; font.pixelSize: 18
                    }
                }
            }

            // Top-down view in the side tile. The live camera opens only
            // from this tile or the Camera button.
            Item {
                id: camTile
                x: 1462; y: 338; width: 300; height: 210
                visible: !app.val("showCamera", false)
                Rectangle {
                    id: camFrame
                    width: parent.width; height: 169
                    radius: 14
                    color: "#070B12"
                    border.color: Qt.rgba(148 / 255, 136 / 255, 224 / 255, 0.35)
                    border.width: 1.5
                    clip: true
                    Canvas {
                        id: miniBev
                        anchors.fill: parent
                        anchors.margins: 2
                        property string spec: app.val("miniBev", "{}")
                        onSpecChanged: requestPaint()
                        onWidthChanged: requestPaint()
                        onHeightChanged: requestPaint()
                        onPaint: {
                            const ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            ctx.fillStyle = "#070B12"
                            ctx.fillRect(0, 0, width, height)
                            let data = {}
                            try { data = JSON.parse(spec || "{}") } catch (e) { data = {} }
                            const lanes = data.lanes || []
                            const cars = data.cars || []
                            const ego = data.ego || 0
                            const rangeM = 70
                            const halfX = 12
                            function px(xm) {
                                return width * 0.5 + (xm / halfX) * (width * 0.42)
                            }
                            function py(ym) {
                                const t = Math.max(0, Math.min(rangeM, ym)) / rangeM
                                return height * 0.90 - t * height * 0.78
                            }
                            ctx.strokeStyle = "rgba(154,154,176,0.28)"
                            ctx.lineWidth = 1
                            ctx.setLineDash([3, 5])
                            for (let r = 20; r <= 60; r += 20) {
                                ctx.beginPath()
                                ctx.arc(px(0), py(0), (r / halfX) * (width * 0.42), Math.PI, 0, false)
                                ctx.stroke()
                                ctx.fillStyle = "rgba(154,154,176,0.7)"
                                ctx.font = "11px sans-serif"
                                ctx.fillText(r + " m", px(0) + 6, py(r) + 4)
                            }
                            ctx.setLineDash([])
                            const leftState = data.left || "normal"
                            const rightState = data.right || "normal"
                            function laneColor(i) {
                                const state = i === 1 ? leftState : (i === 2 ? rightState : "normal")
                                if (state === "warning")
                                    return "#FF5E7E"
                                if (state === "caution")
                                    return "#FFD84D"
                                if (state === "off")
                                    return "#4A4A5C"
                                return "#F4F7FB"
                            }
                            ctx.setLineDash([7, 8])
                            for (let i = 0; i < lanes.length; i++) {
                                const state = i === 1 ? leftState : (i === 2 ? rightState : "normal")
                                ctx.strokeStyle = laneColor(i)
                                ctx.lineWidth = state === "warning" ? 3.5 : 2
                                ctx.beginPath()
                                ctx.moveTo(px(lanes[i]), py(2))
                                ctx.lineTo(px(lanes[i]), py(rangeM))
                                ctx.stroke()
                            }
                            ctx.setLineDash([])
                            function drawCarIcon(cx, cy, color) {
                                ctx.save()
                                ctx.translate(cx, cy)
                                ctx.fillStyle = color
                                ctx.beginPath()
                                ctx.moveTo(0, -12)
                                ctx.quadraticCurveTo(7.5, -10, 7.5, -5)
                                ctx.lineTo(7.5, 7)
                                ctx.quadraticCurveTo(7.5, 12, 4, 12)
                                ctx.lineTo(-4, 12)
                                ctx.quadraticCurveTo(-7.5, 12, -7.5, 7)
                                ctx.lineTo(-7.5, -5)
                                ctx.quadraticCurveTo(-7.5, -10, 0, -12)
                                ctx.closePath()
                                ctx.fill()
                                ctx.fillStyle = "rgba(7,11,18,0.62)"
                                ctx.beginPath()
                                ctx.moveTo(-3.6, -2)
                                ctx.lineTo(3.6, -2)
                                ctx.lineTo(3.1, 5)
                                ctx.lineTo(-3.1, 5)
                                ctx.closePath()
                                ctx.fill()
                                ctx.restore()
                            }
                            for (let i = 0; i < cars.length; i++) {
                                const c = cars[i]
                                const level = c.t | 0
                                const color = level >= 3 ? "#FF5E7E"
                                    : (level === 2 ? "#FFD84D"
                                    : (level === 1 ? "#4DFFB4" : "#8FD4FF"))
                                drawCarIcon(px(c.x), py(c.y), color)
                            }
                            ctx.fillStyle = "#4CD6FF"
                            ctx.beginPath()
                            ctx.moveTo(px(ego), py(0) - 16)
                            ctx.lineTo(px(ego) - 8, py(0) + 6)
                            ctx.lineTo(px(ego) + 8, py(0) + 6)
                            ctx.closePath()
                            ctx.fill()
                            ctx.fillStyle = "#9A9AB0"
                            ctx.font = "12px sans-serif"
                            ctx.fillText("BEV", 10, 18)
                        }
                    }
                }
                Row {
                    anchors.top: camFrame.bottom
                    anchors.topMargin: 12
                    anchors.horizontalCenter: parent.horizontalCenter
                    spacing: 8
                    Rectangle {
                        width: 8; height: 8; radius: 4
                        anchors.verticalCenter: parent.verticalCenter
                        color: app.infoCyan
                        opacity: app.val("paused", false) ? 0.4 : 1
                    }
                    Text {
                        text: "BIRD'S EYE"
                        color: app.textSecondary
                        font.family: app.uiFont; font.pixelSize: 14
                        font.weight: Font.DemiBold; font.letterSpacing: 1.5
                    }
                }
                MouseArea {
                    anchors.fill: camFrame
                    cursorShape: Qt.PointingHandCursor
                    onClicked: cluster.toggleCamera()
                }
            }
        }

        // Back to the 3D view while the camera fills the centre.
        Rectangle {
            visible: app.val("showCamera", false)
            x: 1500; y: 420; width: 220; height: 52; radius: 26
            color: Qt.rgba(18 / 255, 18 / 255, 28 / 255, 0.9)
            border.color: app.infoCyan; border.width: 1.5
            Text {
                anchors.centerIn: parent
                text: "3D view  (C)"
                color: app.infoCyan
                font.family: app.uiFont; font.pixelSize: 18; font.weight: Font.DemiBold
            }
            MouseArea { anchors.fill: parent; onClicked: cluster.toggleCamera() }
        }

        // ---------------------------------------------- left: driving data
        Column {
            x: 280; y: 292; width: 300
            spacing: 0

            // Telltales (ISO 2575-style). Green armed, red active, grey off.
            Row {
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 24
                height: 36
                // FCW
                Shape {
                    id: ttFcwIcon
                    width: 44; height: 34
                    readonly property color c: app.stateColor(app.val("ttFcw", "off"))
                    opacity: app.val("ttFcw", "off") === "active" ? 0.55 + 0.45 * app.pulse : 1
                    ShapePath {
                        strokeWidth: 0; fillColor: ttFcwIcon.c
                        PathSvg { path: "M3 24 V18 L8 11 Q9.5 9 12 9 H22 Q24.5 9 26 11 L30 17 H32 Q35 17 35 20 V24 Z M6.5 26 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 Z M24.5 26 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 Z" }
                    }
                    ShapePath {
                        fillColor: "transparent"; strokeColor: ttFcwIcon.c
                        strokeWidth: 2.6; capStyle: ShapePath.RoundCap
                        PathSvg { path: "M38 9 L42 5 M39 15 H44 M38 21 L42 25" }
                    }
                }
                // LDW
                Shape {
                    id: ttLdwIcon
                    width: 36; height: 32
                    readonly property color c: app.stateColor(app.val("ttLdw", "off"))
                    opacity: app.val("ttLdw", "off") === "active" ? 0.55 + 0.45 * app.pulse : 1
                    ShapePath {
                        fillColor: "transparent"; strokeColor: ttLdwIcon.c
                        strokeWidth: 2.6; capStyle: ShapePath.RoundCap
                        PathSvg { path: "M7 30 L13 2 M29 30 L23 2" }
                    }
                    ShapePath {
                        strokeWidth: 0; fillColor: ttLdwIcon.c
                        PathSvg { path: "M15 14 H21 Q23 14 23 16 V25 Q23 27 21 27 H15 Q13 27 13 25 V16 Q13 14 15 14 Z" }
                    }
                }
                // ISA (sign recognition)
                Rectangle {
                    width: 26; height: 32; radius: 4
                    anchors.verticalCenter: parent.verticalCenter
                    color: "transparent"
                    border.width: 2.4
                    border.color: app.stateColor(app.val("ttIsa", "off"))
                    Rectangle {
                        anchors.horizontalCenter: parent.horizontalCenter
                        y: 7; width: 14; height: 2.4
                        color: parent.border.color
                    }
                    Rectangle {
                        anchors.horizontalCenter: parent.horizontalCenter
                        y: 13; width: 14; height: 10; radius: 2
                        color: parent.border.color
                    }
                }
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                topPadding: 18
                text: app.val("speedText", "--")
                color: "#FFFFFF"
                font.family: app.displayFont
                font.pixelSize: 136
                font.weight: Font.Bold
                font.letterSpacing: -2
                font.features: { "tnum": 1 }
                style: Text.Normal
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                topPadding: 4
                text: app.val("speedUnit", "MPH")
                color: app.textSecondary
                font.family: app.uiFont; font.pixelSize: 28; font.weight: Font.Medium
                font.letterSpacing: 2
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                topPadding: 6
                visible: app.val("speedSourceText", "") !== ""
                text: app.val("speedSourceText", "")
                color: app.textDisabled
                font.family: app.uiFont; font.pixelSize: 15
            }

            Item { width: 1; height: 22 }

            // Function status rows
            Repeater {
                model: [
                    { key: "ttFcw", label: "Forward collision", textKey: "fcwText" },
                    { key: "ttLdw", label: "Lane departure", textKey: "ldwText" }
                ]
                Row {
                    anchors.horizontalCenter: parent.horizontalCenter
                    spacing: 10
                    height: 30
                    Rectangle {
                        width: 9; height: 9; radius: 4.5
                        anchors.verticalCenter: parent.verticalCenter
                        color: app.stateColor(app.val(modelData.key, "off"))
                    }
                    Text {
                        width: 160
                        text: modelData.label
                        color: app.textSecondary
                        font.family: app.uiFont; font.pixelSize: 17
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    Text {
                        width: 92
                        horizontalAlignment: Text.AlignRight
                        text: app.val(modelData.textKey, "Off")
                        color: app.stateColor(app.val(modelData.key, "off"))
                        font.family: app.uiFont; font.pixelSize: 17; font.weight: Font.DemiBold
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }
        }

        // ------------------------------------------------ bottom status bar
        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            y: 728
            spacing: 44
            opacity: app.dimOpacity

            Row {
                spacing: 14
                anchors.verticalCenter: parent.verticalCenter
                Text {
                    text: "TRIP"
                    color: "#FFFFFF"
                    font.family: app.uiFont; font.pixelSize: 22; font.weight: Font.Bold
                    font.letterSpacing: 1
                    anchors.baseline: tripVal.baseline
                }
                Text {
                    id: tripVal
                    text: app.val("tripText", "—")
                    color: "#FFFFFF"
                    font.family: app.uiFont; font.pixelSize: 26; font.weight: Font.Medium
                    font.features: { "tnum": 1 }
                }
                Text {
                    text: app.val("tripUnit", "")
                    color: app.textSecondary
                    anchors.baseline: tripVal.baseline
                    font.family: app.uiFont; font.pixelSize: 20
                }
            }

            // Engine chips: which perception models are running.
            Row {
                spacing: 18
                anchors.verticalCenter: parent.verticalCenter
                Repeater {
                    model: [
                        { k: "engLanes", t: "LANES" },
                        { k: "engVehicles", t: "VEHICLES" },
                        { k: "engSigns", t: "SIGNS" }
                    ]
                    Row {
                        spacing: 7
                        Rectangle {
                            width: 8; height: 8; radius: 4
                            anchors.verticalCenter: parent.verticalCenter
                            color: app.val(modelData.k, false) ? app.infoCyan : app.textDisabled
                        }
                        Text {
                            text: modelData.t
                            color: app.val(modelData.k, false) ? "#FFFFFF" : app.textDisabled
                            font.family: app.uiFont; font.pixelSize: 17
                            font.weight: Font.DemiBold; font.letterSpacing: 1
                        }
                    }
                }
            }

            Row {
                spacing: 6
                anchors.verticalCenter: parent.verticalCenter
                Text {
                    id: fpsVal
                    text: app.val("fpsText", "—")
                    color: "#FFFFFF"
                    font.family: app.uiFont; font.pixelSize: 26; font.weight: Font.Medium
                    font.features: { "tnum": 1 }
                }
                Text {
                    text: "FPS"
                    color: app.textSecondary
                    anchors.baseline: fpsVal.baseline
                    font.family: app.uiFont; font.pixelSize: 20
                }
            }
            Row {
                spacing: 6
                anchors.verticalCenter: parent.verticalCenter
                Text {
                    id: latVal
                    text: app.val("latencyText", "—")
                    color: "#FFFFFF"
                    font.family: app.uiFont; font.pixelSize: 26; font.weight: Font.Medium
                    font.features: { "tnum": 1 }
                }
                Text {
                    text: "ms"
                    color: app.textSecondary
                    anchors.baseline: latVal.baseline
                    font.family: app.uiFont; font.pixelSize: 20
                }
            }
            Text {
                visible: app.val("sourceKind", "") === "sim"
                text: "SIMULATION"
                color: app.violet
                anchors.verticalCenter: parent.verticalCenter
                font.family: app.uiFont; font.pixelSize: 20; font.weight: Font.Bold
                font.letterSpacing: 1
            }
        }

        // -------------------------------- operator controls (below bezel)
        Item {
            id: controls
            x: 0; y: 860; width: 1920; height: 100
            opacity: ctlHover.hovered ? 1.0 : 0.85
            z: 20
            Behavior on opacity { NumberAnimation { duration: 180 } }
            HoverHandler { id: ctlHover }

            Row {
                anchors.centerIn: parent
                spacing: 14
                Repeater {
                    model: [
                        { a: "pause", t: app.val("paused", false) ? "Play  ␣" : "Pause  ␣" },
                        { a: "open", t: "Open video  O" },
                        { a: "camera", t: app.val("showCamera", false) ? "3D view  C" : "Camera  C" },
                        { a: "lanes", t: app.val("laneDraw", true) ? "Lane paint on  L" : "Lane paint off  L" },
                        { a: "mute", t: app.val("muted", false) ? "Sound off  M" : "Sound on  M" },
                        { a: "full", t: "Full screen  F" }
                    ]
                    Rectangle {
                        width: ctlText.implicitWidth + 32
                        height: 40
                        radius: 20
                        color: ctlMouse.containsMouse ? "#1C1B2C" : "#12121C"
                        border.color: Qt.rgba(148 / 255, 136 / 255, 224 / 255, 0.4)
                        border.width: 1
                        Text {
                            id: ctlText
                            anchors.centerIn: parent
                            text: modelData.t
                            color: "#C9C9DA"
                            font.family: app.uiFont; font.pixelSize: 16; font.weight: Font.Medium
                        }
                        MouseArea {
                            id: ctlMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: app.action(modelData.a)
                        }
                    }
                }
            }
        }
    }

    // ============================================================ actions
    function action(a) {
        if (a === "pause") cluster.togglePause()
        else if (a === "open") videoDialog.open()
        else if (a === "camera") cluster.toggleCamera()
        else if (a === "lanes") cluster.toggleLaneDraw()
        else if (a === "mute") cluster.toggleMute()
        else if (a === "full") cluster.toggleFullscreen()
    }

    FileDialog {
        id: videoDialog
        title: "Open video"
        nameFilters: ["Video files (*.mp4 *.avi *.mkv *.mov)", "All files (*)"]
        onAccepted: cluster.openVideo(selectedFile.toString())
    }

    focus: true
    Keys.onPressed: function (event) {
        if (event.isAutoRepeat)
            return
        switch (event.key) {
        case Qt.Key_Space: app.action("pause"); break
        case Qt.Key_O: app.action("open"); break
        case Qt.Key_C: app.action("camera"); break
        case Qt.Key_L: app.action("lanes"); break
        case Qt.Key_M: app.action("mute"); break
        case Qt.Key_F: app.action("full"); break
        case Qt.Key_Escape: cluster.exitFullscreen(); break
        case Qt.Key_Q: cluster.quit(); break
        default: return
        }
        event.accepted = true
    }
}
