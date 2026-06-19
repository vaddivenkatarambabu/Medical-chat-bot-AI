export function Blobs() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      <div
        className="blob animate-blob h-[420px] w-[420px] -left-32 -top-32"
        style={{ background: "var(--gradient-primary)" }}
      />
      <div
        className="blob animate-blob h-[380px] w-[380px] right-0 top-1/3"
        style={{ background: "var(--primary-glow)", animationDelay: "-6s" }}
      />
      <div
        className="blob animate-blob h-[340px] w-[340px] bottom-0 left-1/3"
        style={{ background: "var(--accent)", animationDelay: "-12s" }}
      />
    </div>
  );
}
