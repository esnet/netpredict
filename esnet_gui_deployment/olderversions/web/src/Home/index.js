import React, { useState } from 'react';
import Home from './Home';
import mapData from '../map-data.json';

const predictionsData = [
  ['2017-01-24T00:00', 0.01, ['FNAL', 'STAR']],
  ['2017-01-24T01:00', 0.13, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T02:00', 0.07, ['SACR', 'DENV']],
  ['2017-01-24T03:00', 0.04, ['FNAL', 'STAR']],
  ['2017-01-24T04:00', 0.33, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T05:00', 0, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T06:00', 0, ['FNAL', 'STAR']],
  ['2017-01-24T07:00', 0, ['SACR', 'DENV']],
  ['2017-01-24T08:00', 0.95, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T09:00', 1.12, ['FNAL', 'STAR']],
  ['2017-01-24T10:00', 0.66, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T11:00', 0.06, ['SACR', 'DENV']],
  ['2017-01-24T12:00', 0.3, ['FNAL', 'STAR']],
  ['2017-01-24T13:00', 0.05, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T14:00', 0.5, ['SACR', 'DENV']],
  ['2017-01-24T15:00', 0.24, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T16:00', 0.02, ['SACR', 'DENV']],
  ['2017-01-24T17:00', 0.98, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T18:00', 0.46, ['SACR', 'DENV']],
  ['2017-01-24T19:00', 0.8, ['FNAL', 'STAR']],
  ['2017-01-24T20:00', 0.39, ['SACR', 'DENV']],
  ['2017-01-24T21:00', 0.4, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T22:00', 0.39, ['NREL', 'DENV', 'KANS']],
  ['2017-01-24T23:00', 0.28, ['SACR', 'DENV']],
];

const sites = [
  'ALBQ',
  'AMES',
  'AMST',
  'ANL',
  'AOFA',
  'ATLA',
  'BNL',
  'BOIS',
  'BOST',
  'CERN',
  'CERN513',
  'CERN773',
  'CHIC',
  'DENV',
  'ELPA',
  'EQX-ASH',
  'EQX-CHI',
  'ETTP',
  'FNAL',
  'GA',
  'HOUS',
  'INL',
  'JGI',
  'JLAB',
  'KANS',
  'KCNSC',
  'KCNSC-NM',
  'LANL',
  'LBNL',
  'LIGO',
  'LLNL',
  'LNS',
  'LOND',
  'LSVN',
  'NASH',
  'NERSC',
  'NETL-MGN',
  'NETL-PGH',
  'NEWY',
  'NGA-SW',
  'NNSS',
  'NPS',
  'NREL',
  'ORAU',
  'ORNL',
  'PANTEX',
  'PNNL',
  'PNWG',
  'PPPL',
  'PSFC',
  'SACR',
  'SLAC',
  'SNLA',
  'SNLL',
  'SRS',
  'STAR',
  'SUNN',
  'WASH',
  'Y12',
];

export default function HomeContainer() {
  const [predictions, setPredictions] = useState(null);
  const [loadingPredictions, setLoadingPredictions] = useState(false);

  function loadPredictions() {
    setPredictions(predictionsData);
  }

  return <Home predictions={predictions} sites={sites} mapData={mapData} loadPredictions={loadPredictions} />;
}
