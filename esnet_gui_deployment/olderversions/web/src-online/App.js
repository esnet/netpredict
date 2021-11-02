import { Index, TimeSeries } from "pondjs";
import React from "react";
import { Col, Container, Row } from "react-bootstrap";
import { TrafficMap } from "react-network-diagrams";
import Select from "react-select";
import {
    BarChart,
    ChartContainer,
    ChartRow,
    Charts,
    Resizable,
    styler,
    YAxis
} from "react-timeseries-charts";
import data from "./map-data.json";

const predictions = [
    ["2017-01-24T00:00", 0.01],
    ["2017-01-24T01:00", 0.13],
    ["2017-01-24T02:00", 0.07],
    ["2017-01-24T03:00", 0.04],
    ["2017-01-24T04:00", 0.33],
    ["2017-01-24T05:00", 0],
    ["2017-01-24T06:00", 0],
    ["2017-01-24T07:00", 0],
    ["2017-01-24T08:00", 0.95],
    ["2017-01-24T09:00", 1.12],
    ["2017-01-24T10:00", 0.66],
    ["2017-01-24T11:00", 0.06],
    ["2017-01-24T12:00", 0.3],
    ["2017-01-24T13:00", 0.05],
    ["2017-01-24T14:00", 0.5],
    ["2017-01-24T15:00", 0.24],
    ["2017-01-24T16:00", 0.02],
    ["2017-01-24T17:00", 0.98],
    ["2017-01-24T18:00", 0.46],
    ["2017-01-24T19:00", 0.8],
    ["2017-01-24T20:00", 0.39],
    ["2017-01-24T21:00", 0.4],
    ["2017-01-24T22:00", 0.39],
    ["2017-01-24T23:00", 0.28]
];

const series = new TimeSeries({
    name: "predictions",
    columns: ["index", "value"],
    points: predictions.map(([d, value]) => [Index.getIndexString("1h", new Date(d)), value])
});

const sites = [
    "ALBQ",
    "AMES",
    "AMST",
    "ANL",
    "AOFA",
    "ATLA",
    "BNL",
    "BOIS",
    "BOST",
    "CERN",
    "CERN513",
    "CERN773",
    "CHIC",
    "DENV",
    "ELPA",
    "EQX-ASH",
    "EQX-CHI",
    "ETTP",
    "FNAL",
    "GA",
    "HOUS",
    "INL",
    "JGI",
    "JLAB",
    "KANS",
    "KCNSC",
    "KCNSC-NM",
    "LANL",
    "LBNL",
    "LIGO",
    "LLNL",
    "LNS",
    "LOND",
    "LSVN",
    "NASH",
    "NERSC",
    "NETL-MGN",
    "NETL-PGH",
    "NEWY",
    "NGA-SW",
    "NNSS",
    "NPS",
    "NREL",
    "ORAU",
    "ORNL",
    "PANTEX",
    "PNNL",
    "PNWG",
    "PPPL",
    "PSFC",
    "SACR",
    "SLAC",
    "SNLA",
    "SNLL",
    "SRS",
    "STAR",
    "SUNN",
    "WASH",
    "Y12"
];

export default class Map extends React.Component {
    state = {
        source: null,
        dest: null
    };

    handleSourceChange(selected) {
        this.setState({ source: selected });
    }

    handleDestChange(selected) {
        this.setState({ dest: selected });
    }

    render() {
        const nodes = data.data.mapTopology.nodes.map(n => {
            const { x, y, ...other } = n;
            return {
                type: "node",
                x: x,
                y: y * 1.2,
                ...other
            };
        });
        const edges = data.data.mapTopology.edges.map(e => {
            const { name, ...other } = e;
            const parts = name.split("--");
            const source = parts[0];
            const target = parts[1];
            return {
                source,
                target,
                ...other
            };
        });

        const topology = {
            nodes,
            edges
        };

        const bounds = {
            x1: -5,
            y1: 5,
            x2: 240,
            y2: 140
        };

        const nodeSizeMap = {
            node: 4
        };

        const edgeThicknessMap = {
            "100G": 3,
            "10G": 3,
            "1G": 3,
            subG: 3
        };

        // A style to use for nodes and their labels
        const nodeSyleMap = {
            node: {
                normal: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    cursor: "pointer"
                },
                selected: {
                    fill: "#37B6D3",
                    stroke: "rgba(55, 182, 211, 0.22)",
                    strokeWidth: 10,
                    cursor: "pointer"
                },
                muted: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    opacity: 0.6,
                    cursor: "pointer"
                }
            },
            label: {
                normal: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 9
                },
                selected: {
                    fill: "#333",
                    stroke: "none",
                    fontSize: 11
                },
                muted: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 8
                }
            }
        };

        // Mapping of node type to style
        const stylesMap = {
            node: nodeSyleMap
        };

        //
        // Source and desitintion selection

        const options = sites.map(site => {
            return {
                value: site,
                label: site
            };
        });

        const labelStyle = {
            fontSize: 12,
            color: "#989898",
            paddingLeft: 4
        };

        const toolbar = (
            <div>
                <div style={labelStyle}>SOURCE:</div>
                <div style={{ width: 250, padding: 3 }}>
                    <Select
                        options={options}
                        value={this.state.source}
                        onChange={s => this.handleSourceChange(s)}
                    />
                </div>
                <div style={labelStyle}>DEST:</div>
                <div style={{ width: 250, padding: 3 }}>
                    <Select
                        options={options}
                        value={this.state.dest}
                        onChange={s => this.handleDestChange(s)}
                    />
                </div>
            </div>
        );

        const style = styler([{ key: "value", color: "#AAA", selected: "#2CB1CF" }]);

        const chart = (
            <Resizable>
                <ChartContainer timeRange={series.range()}>
                    <ChartRow height="120">
                        <YAxis
                            id="value"
                            label="Prediction"
                            min={0}
                            max={1.5}
                            format=".2f"
                            width="70"
                            type="linear"
                        />
                        <Charts>
                            <BarChart
                                axis="value"
                                style={style}
                                spacing={1}
                                columns={["value"]}
                                series={series}
                                minBarHeight={1}
                            />
                        </Charts>
                    </ChartRow>
                </ChartContainer>
            </Resizable>
        );

        return (
            <Container fluid={true} style={{ background: "white" }}>
                <Row>
                    <Col md={12} style={{ padding: 0 }}>
                        <div
                            style={{
                                backgroundColor: "white",
                                color: "black",
                                height: 50
                            }}
                        >
                            <h1 style={{ color: "#777", marginLeft: 20 }}>Mouse Trap</h1>
                        </div>
                    </Col>
                </Row>
                <Row
                    style={{
                        backgroundColor: "white",
                        borderTopStyle: "solid",
                        borderWidth: 1,
                        borderColor: "#DDD",
                        borderBottomStyle: "solid"
                    }}
                >
                    <Col
                        md={3}
                        style={{
                            height: 160,
                            paddingTop: 10,
                            background: "white",
                            borderWidth: 1,
                            borderColor: "#DDD",
                            borderRightStyle: "solid"
                        }}
                    >
                        {toolbar}
                    </Col>
                    <Col md={9} style={{ paddingTop: 3, paddingRight: 10 }}>
                        {chart}
                    </Col>
                </Row>
                <Row>
                    <Col md={12} style={{ padding: 0 }}>
                        <div
                            style={{
                                borderWidth: 1,
                                borderColor: "#DDD",
                                borderBottomStyle: "solid"
                            }}
                        >
                            <TrafficMap
                                style={{ background: "#f2f5f7" }}
                                topology={topology}
                                height={300}
                                bounds={bounds}
                                edgeDrawingMethod="simple"
                                nodeSizeMap={nodeSizeMap}
                                edgeThicknessMap={edgeThicknessMap}
                                stylesMap={stylesMap}
                            />
                        </div>
                    </Col>
                </Row>
            </Container>
        );
    }
}
