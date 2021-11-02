import React from 'react';
import { TrafficMap } from 'react-network-diagrams';

export default function ({ mapData, path }) {
    
    const nodes = mapData.data.mapTopology.nodes.map((n) => {
        const { x, y, ...other } = n;
        return {
          type: 'node',
          x: x,
          y: y * 0.8,
          ...other,
        };
      });
      const edges = mapData.data.mapTopology.edges.map((e) => {
        const { name, ...other } = e;
        const parts = name.split('--');
        const source = parts[0];
        const target = parts[1];
        return {
          source,
          target,
          color: 'gray',
          ...other,
        };
      });
  
      const topology = {
        nodes,
        edges,
      };

      if (path) {
        topology.paths = [
            {
              name: 'Path',
              steps: path,
            },
          ];
      }
  
      const bounds = {
        x1: -10,
        y1: -5,
        x2: 245,
        y2: 120,
      };
  
      const nodeSizeMap = {
        node: 4,
      };
  
      const edgeThicknessMap = {
        '100G': 3,
        '10G': 3,
        '1G': 3,
        subG: 3,
      };
  
      // A style to use for nodes and their labels
      const nodeSyleMap = {
        node: {
          normal: {
            fill: '#696969',
            stroke: '#696969',
          },
          selected: {
            fill: '#37B6D3',
            stroke: 'rgba(55, 182, 211, 0.22)',
            strokeWidth: 10,
            cursor: 'pointer',
          },
          muted: {
            fill: '#CBCBCB',
            stroke: '#BEBEBE',
            opacity: 0.6,
            cursor: 'pointer',
          },
        },
        label: {
          normal: {
            fill: '#696969',
            stroke: 'none',
            fontSize: 9,
          },
          selected: {
            fill: '#333',
            stroke: 'none',
            fontSize: 11,
          },
          muted: {
            fill: '#696969',
            stroke: 'none',
            fontSize: 8,
          },
        },
      };
  
      // Mapping of node type to style
      const stylesMap = {
        node: nodeSyleMap,
      };

      const background = require('../images/NetPredict-05.png');

      return (
        <div style={{ backgroundImage: `url(${background})` }} className="traffic-map-container">
          <TrafficMap
            style={{ background: '#fff' }}
            topology={topology}
            showPaths={true}
            bounds={bounds}
            edgeDrawingMethod="simple"
            nodeSizeMap={nodeSizeMap}
            edgeThicknessMap={edgeThicknessMap}
            stylesMap={stylesMap}
        />
        </div>
        
      );
}