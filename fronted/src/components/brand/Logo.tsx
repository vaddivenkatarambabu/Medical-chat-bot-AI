import logo from "@/assets/medicore-logo.png";
import { cn } from "@/lib/utils";

export function Logo({
  className,
  withText = true,
  size = 32,
}: {
  className?: string;
  withText?: boolean;
  size?: number;
}) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <img
        src={logo}
        alt="MediCore logo"
        width={size}
        height={size}
        className="rounded-lg shadow-[var(--shadow-elegant)]"
      />
      {withText && (
        <span className="font-display text-lg font-bold tracking-tight">
          MediCore
        </span>
      )}
    </div>
  );
}
