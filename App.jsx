/*
Task 3: React Dashboard
========================
Place at: frontend/src/App.jsx

Setup (from your frontend/ folder):
    npm create vite@latest . -- --template react
    npm install recharts
    npm run dev

Assumes the Flask backend (backend/app.py) is running on http://localhost:5000
*/

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const API_BASE = "http://localhost:5000/api";

export default function App() {
  const [prices, setPrices] = useState([]);
  const [events, setEvents] = useState([]);
  const [changePoint, setChangePoint] = useState(null);
  const [dateRange, setDateRange] = useState({ start: "", end: "" });
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadAll() {
      try {
        setLoading(true);
        const [pricesRes, eventsRes, cpRes] = await Promise.all([
          fetch(`${API_BASE}/prices`),
          fetch(`${API_BASE}/events`),
          fetch(`${API_BASE}/change-points`),
        ]);
        if (!pricesRes.ok || !eventsRes.ok) throw new Error("Failed to load data");

        setPrices(await pricesRes.json());
        setEvents(await eventsRes.json());
        setChangePoint(cpRes.ok ? await cpRes.json() : null);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadAll();
  }, []);

  const filteredPrices = useMemo(() => {
    return prices.filter((p) => {
      if (dateRange.start && p.date < dateRange.start) return false;
      if (dateRange.end && p.date > dateRange.end) return false;
      return true;
    });
  }, [prices, dateRange]);

  const visibleEvents = useMemo(() => {
    return events.filter((e) => {
      if (dateRange.start && e.Date < dateRange.start) return false;
      if (dateRange.end && e.Date > dateRange.end) return false;
      return true;
    });
  }, [events, dateRange]);

  if (loading) return <div style={styles.center}>Loading dashboard…</div>;
  if (error)
    return (
      <div style={styles.center}>
        Could not reach the backend at {API_BASE}. Is Flask running? ({error})
      </div>
    );

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={{ margin: 0 }}>Brent Oil Price — Change Point Dashboard</h1>
        <p style={{ margin: "4px 0 0", color: "#666" }}>
          Birhan Energies · Bayesian change point analysis, 1987–2022
        </p>
      </header>

      <section style={styles.filters}>
        <label>
          Start date:
          <input
            type="date"
            value={dateRange.start}
            onChange={(e) => setDateRange((r) => ({ ...r, start: e.target.value }))}
            style={styles.input}
          />
        </label>
        <label>
          End date:
          <input
            type="date"
            value={dateRange.end}
            onChange={(e) => setDateRange((r) => ({ ...r, end: e.target.value }))}
            style={styles.input}
          />
        </label>
        <button style={styles.resetBtn} onClick={() => setDateRange({ start: "", end: "" })}>
          Reset range
        </button>
      </section>

      {changePoint && (
        <section style={styles.card}>
          <h3 style={{ marginTop: 0 }}>Detected Change Point</h3>
          <p>
            <strong>{changePoint.change_point_date}</strong> — average price shifted from{" "}
            <strong>${changePoint.avg_price_before_usd}</strong> to{" "}
            <strong>${changePoint.avg_price_after_usd}</strong> (
            {changePoint.price_pct_change > 0 ? "+" : ""}
            {changePoint.price_pct_change}%). Model confidence:{" "}
            {(changePoint.prob_increase_after * 100).toFixed(1)}% probability of an increase
            after the break.
          </p>
        </section>
      )}

      <section style={{ ...styles.card, height: 420 }}>
        <h3 style={{ marginTop: 0 }}>Price History with Event Markers</h3>
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={filteredPrices} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={60} />
            <YAxis domain={["auto", "auto"]} label={{ value: "USD/barrel", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="price" stroke="#1565c0" dot={false} name="Brent price" />
            {changePoint && (
              <ReferenceLine
                x={changePoint.change_point_date}
                stroke="#c62828"
                strokeWidth={2}
                label={{ value: "Change point", position: "top", fill: "#c62828" }}
              />
            )}
            {visibleEvents.map((ev) => (
              <ReferenceLine
                key={ev.Date + ev.event_name}
                x={ev.Date}
                stroke="#9e9e9e"
                strokeDasharray="4 4"
                onClick={() => setSelectedEvent(ev)}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section style={styles.card}>
        <h3 style={{ marginTop: 0 }}>Events in Range ({visibleEvents.length})</h3>
        <div style={{ maxHeight: 260, overflowY: "auto" }}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Event</th>
                <th>Category</th>
                <th>Expected direction</th>
              </tr>
            </thead>
            <tbody>
              {visibleEvents.map((ev) => (
                <tr
                  key={ev.Date + ev.event_name}
                  style={{
                    cursor: "pointer",
                    background: selectedEvent?.Date === ev.Date ? "#e3f2fd" : "transparent",
                  }}
                  onClick={() => setSelectedEvent(ev)}
                >
                  <td>{ev.Date}</td>
                  <td>{ev.event_name || ev.description}</td>
                  <td>{ev.category}</td>
                  <td>{ev.expected_direction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selectedEvent && (
          <div style={{ marginTop: 12, padding: 12, background: "#f5f5f5", borderRadius: 6 }}>
            <strong>{selectedEvent.event_name}</strong> — {selectedEvent.description}
          </div>
        )}
      </section>
    </div>
  );
}

const styles = {
  page: { fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" },
  header: { marginBottom: 20 },
  filters: { display: "flex", gap: 16, alignItems: "center", marginBottom: 20, flexWrap: "wrap" },
  input: { marginLeft: 8, padding: "4px 8px" },
  resetBtn: { padding: "6px 12px", cursor: "pointer" },
  card: {
    background: "#fff",
    border: "1px solid #e0e0e0",
    borderRadius: 8,
    padding: 20,
    marginBottom: 20,
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
  },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 14 },
  center: { display: "flex", height: "100vh", alignItems: "center", justifyContent: "center", fontFamily: "sans-serif" },
};
