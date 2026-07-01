import { useMemo, useState } from 'react';
import { AreaChart, Area, BarChart, Bar, CartesianGrid, Cell, LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useLiveTelemetry } from './hooks/useLiveTelemetry';
import { MetricCard } from './components/MetricCard';
import { Pill } from './components/Pill';
import { Panel } from './components/Panel';

export default function App() {
  const { status, packet, series, alerts } = useLiveTelemetry();
  const trendData = useMemo(() => series.slice(-36), [series]);
  const alertData = useMemo(() => alerts.slice(0, 6), [alerts]);

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Solar storm live monitor</p>
          <h1>GOES / WIND real-time dashboard</h1>
          <p className="lede">
            Live replay now, real Socket.IO later. This monitor is designed to track solar wind trends, derived risk,
            and alert activity in a dark glass operational UI.
          </p>
        </div>
        <Pill tone={status === 'live' ? 'green' : status === 'connecting' ? 'amber' : 'slate'}>
          {status}
        </Pill>
      </header>

      <section className="metrics-grid">
        <MetricCard label="Bz / Velocity risk" value={packet.risk.toFixed(2)} detail={`Packet ${packet.sequence}`} />
        <MetricCard label="Solar wind speed" value={`${packet.speed} km/s`} detail={`Density ${packet.density} p/cc`} />
        <MetricCard label="IMF deviation" value={`${packet.imf} nT`} detail={`Kp proxy ${packet.kp}`} />
        <MetricCard label="Active alerts" value={String(alertData.length)} detail="Rolling alert window" />
      </section>

      <section className="grid two-col">
        <Panel title="Risk trend" subtitle="Rolling derived risk over the latest packets">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.45} />
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="4 4" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 12 }} domain={[0, 1]} />
              <Tooltip />
              <Area type="monotone" dataKey="risk" stroke="#38bdf8" fill="url(#riskFill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Telemetry snapshot" subtitle="Current packet composition">
          <div className="snapshot-list">
            <div><span>Timestamp</span><strong>{packet.timestamp}</strong></div>
            <div><span>GOES</span><strong>{packet.goes}</strong></div>
            <div><span>WIND</span><strong>{packet.wind}</strong></div>
            <div><span>Alert state</span><strong>{packet.alert}</strong></div>
          </div>
          <div className="bar-wrap">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={trendData.slice(-10)}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="4 4" />
                <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 12 }} domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="risk" radius={[8, 8, 0, 0]}>
                  {trendData.slice(-10).map((entry, index) => (
                    <Cell key={entry.label} fill={index % 2 === 0 ? '#38bdf8' : '#0ea5e9'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </section>

      <section className="grid two-col bottom-grid">
        <Panel title="Alert feed" subtitle="Most recent risk events">
          <div className="alert-list">
            {alertData.map((alert) => (
              <div key={alert.id} className={`alert-card ${alert.level}`}>
                <div>
                  <strong>{alert.title}</strong>
                  <p>{alert.message}</p>
                </div>
                <span>{alert.time}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Live stream status" subtitle="Ready for Socket.IO backend wiring">
          <div className="status-card">
            <p><span>Connection</span> {status}</p>
            <p><span>Last packet</span> {packet.timestamp}</p>
            <p><span>Source</span> {status === 'live' ? 'Socket.IO backend' : 'Historical replay'}</p>
          </div>
        </Panel>
      </section>
    </div>
  );
}
