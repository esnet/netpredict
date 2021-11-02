import React from 'react';
import { Index, TimeSeries } from 'pondjs';
import {
    BarChart,
    ChartContainer,
    ChartRow,
    Charts,
    Resizable,
    styler,
    YAxis,
  } from 'react-timeseries-charts';
  import moment from 'moment';


export default function ({ predictions, mapData, highlight, selection, handleBarHighlight, handleBarSelection }) {
    if (predictions === null || mapData === null) {
        return null;
    }

    const series = new TimeSeries({
        name: 'predictions',
        columns: ['index', 'value', 'path'],
        points: predictions.map(([d, value, path]) => [
          Index.getIndexString('1h', new Date(d)),
          value,
          path,
        ]),
      });

      const barChartStyle = styler([
        {
          key: 'value',
          color: '#7eaebf',
          selected: '#0eaada',
          highlighted: '#7F4C9E',
          muted: '#CCC',
        },
      ]);

    let labelText;
    let infoValues = null;

    if (highlight) {
      labelText = `${highlight.event.get('value')}TB (Good)`;
      infoValues = [{ label: 'Prediction', value: labelText }];
    }

    return (
        <Resizable>
          <ChartContainer
            timeRange={series.range()}
            onBackgroundClick={() => handleBarSelection(null)}
          >
            <ChartRow height="175">
              <YAxis
                id="value"
                label="Size (Terabytes)"
                min={0}
                max={1.5}
                format=".2f"
                width="70"
                type="linear"
              />
              <Charts>
                <BarChart
                  axis="value"
                  style={barChartStyle}
                  spacing={1}
                  columns={['value']}
                  series={series}
                  info={infoValues}
                  infoWidth={150}
                  infoTimeFormat={(index) => {
                    return moment(index.begin()).format('hh A');
                  }}
                  minBarHeight={3}
                  highlighted={highlight}
                  onHighlightChange={handleBarHighlight}
                  selected={selection}
                  onSelectionChange={handleBarSelection}
                />
              </Charts>
              <YAxis
                id="value1"
                min={0}
                max={1.5}
                format=".2f"
                width="40"
                type="linear"
              />
            </ChartRow>
          </ChartContainer>
        </Resizable>
      );
}