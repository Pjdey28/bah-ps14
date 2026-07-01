import { useEffect, useMemo, useRef, useState } from 'react';
import { io } from 'socket.io-client';

const fallbackSeries = Array.from({ length: 48 }, (_, index) => {
  const risk = Number((0.2 + Math.random() * 0.6 + (index % 7) * 0.02).toFixed(2));
  return {
    label: `T-${48 - index}`,
    risk,
  };
});

const fallbackAlerts = [
  { id: 1, title: 'Geomagnetic watch', message: 'Bz trend drifted negative.', level: 'watch', time: '2 min ago' },
  { id: 2, title: 'Telemetry spike', message: 'Velocity increased above baseline.', level: 'watch', time: '5 min ago' },
  { id: 3, title: 'Moderate alert', message: 'Risk score crossed the medium threshold.', level: 'medium', time: '9 min ago' },
];

const socketUrl = import.meta.env.VITE_SOCKET_URL || '';

function mapPacket(packet) {
  const risk = Number(packet?.risk ?? packet?.risk_score ?? 0).toFixed(2);

  return {
    timestamp: packet?.timestamp || new Date().toLocaleTimeString(),
    goes: packet?.goes || packet?.source || 'GOES-16',
    wind: packet?.wind || 'WIND',
    risk: Number(risk),
    speed: Number(packet?.speed ?? packet?.solar_wind_speed ?? 0),
    density: Number(packet?.density ?? packet?.proton_density ?? 0),
    imf: Number(packet?.imf ?? packet?.bz ?? 0),
    kp: Number(packet?.kp ?? packet?.kp_proxy ?? 0),
    alert: packet?.alert || packet?.alert_level || 'WATCH',
    sequence: Number(packet?.sequence ?? packet?.packet_id ?? 0),
  };
}

export function useLiveTelemetry() {
  const socketRef = useRef(null);
  const [status, setStatus] = useState(socketUrl ? 'connecting' : 'replay');
  const [packet, setPacket] = useState({
    timestamp: '2026-07-01 00:00',
    goes: 'GOES-16',
    wind: 'WIND',
    risk: 0.42,
    speed: 412,
    density: 4.8,
    imf: 5.9,
    kp: 3,
    alert: 'WATCH',
    sequence: 128,
  });
  const [series, setSeries] = useState(fallbackSeries);
  const [alerts, setAlerts] = useState(fallbackAlerts);

  useEffect(() => {
    if (socketUrl) {
      const socket = io(socketUrl, {
        transports: ['websocket'],
        withCredentials: true,
      });

      socketRef.current = socket;

      socket.on('connect', () => setStatus('live'));
      socket.on('disconnect', () => setStatus('offline'));
      socket.on('connect_error', () => setStatus('offline'));
      socket.on('snapshot', (payload) => {
        if (payload?.packet) {
          setPacket(mapPacket(payload.packet));
        }
        if (Array.isArray(payload?.series)) {
          setSeries(payload.series.map((entry, index) => ({
            label: entry.label || `Pkt ${index + 1}`,
            risk: Number(entry.risk ?? 0),
          })));
        }
        if (Array.isArray(payload?.alerts)) {
          setAlerts(payload.alerts.map((item, index) => ({
            id: item.id ?? index,
            title: item.title ?? 'Alert',
            message: item.message ?? '',
            level: item.level ?? 'watch',
            time: item.time ?? 'now',
          })));
        }
      });
      socket.on('packet', (payload) => {
        const next = mapPacket(payload);
        setPacket(next);
        setSeries((history) => [...history.slice(-47), { label: `Pkt ${next.sequence}`, risk: next.risk }]);
        if (next.risk > 0.72) {
          setAlerts((currentAlerts) => [
            {
              id: next.sequence,
              title: 'High solar storm alert',
              message: 'Risk rose above the critical threshold.',
              level: 'high',
              time: 'just now',
            },
            ...currentAlerts,
          ].slice(0, 6));
        }
      });

      return () => socket.disconnect();
    }

    const timer = window.setInterval(() => {
      setPacket((current) => {
        const nextRisk = Number(Math.min(1, Math.max(0, current.risk + (Math.random() - 0.5) * 0.08)).toFixed(2));
        const next = {
          ...current,
          risk: nextRisk,
          speed: Math.round(380 + Math.random() * 220),
          density: Number((2.5 + Math.random() * 6).toFixed(1)),
          imf: Number((3.5 + Math.random() * 8).toFixed(1)),
          kp: Math.min(9, Math.max(1, Math.round(2 + Math.random() * 5))),
          timestamp: new Date().toLocaleTimeString(),
          sequence: current.sequence + 1,
          alert: nextRisk > 0.72 ? 'ALERT' : nextRisk > 0.48 ? 'WATCH' : 'NORMAL',
        };

        setSeries((history) => [...history.slice(-47), { label: `Pkt ${next.sequence}`, risk: next.risk }]);
        if (nextRisk > 0.72) {
          setAlerts((currentAlerts) => [
            {
              id: next.sequence,
              title: 'High solar storm alert',
              message: 'Risk rose above the critical threshold.',
              level: 'high',
              time: 'just now',
            },
            ...currentAlerts,
          ].slice(0, 6));
        }

        return next;
      });
    }, 1800);

    return () => window.clearInterval(timer);
  }, []);

  const liveStatus = useMemo(() => status, [status]);

  return {
    status: liveStatus,
    packet,
    series,
    alerts,
  };
}
