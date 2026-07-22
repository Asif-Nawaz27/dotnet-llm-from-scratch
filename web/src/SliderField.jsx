// A labeled range slider with a live numeric readout - used for every tunable
// parameter (temperature, top-k, steps, ...) so they all look and behave the same.
export default function SliderField({ label, value, onChange, min, max, step = 1, disabled = false }) {
  return (
    <label className="slider-field">
      <span className="slider-label">
        <span>{label}</span>
        <span className="slider-value">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
