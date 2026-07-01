export function Pill({ tone = 'slate', children }) {
  return <div className={`pill ${tone}`}>{children}</div>;
}
