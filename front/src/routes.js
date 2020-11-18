import React from 'react';

const Dashboard = React.lazy(() => import('./views/Dashboard'));
const RealPrediction = React.lazy(() => import('./pages/RealPrediction'));
const DataPrediction = React.lazy(() => import('./pages/DataPrediction'));

const routes = [
  { path: '/', exact: true, name: 'Home' },
  { path: '/real/prediction', exact: true, name: 'RealPrediction', component: RealPrediction },
  { path: '/data/prediction', exact: true, name: 'DataPrediction', component: DataPrediction },
  // { path: '/dashboard', name: 'Dashboard', component: RealPrediction },
];

export default routes;
