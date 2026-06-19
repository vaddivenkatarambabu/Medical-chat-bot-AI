import { AlertTriangle } from "lucide-react";

export function MedicalDisclaimer() {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-xs text-amber-700 dark:text-amber-300">
      <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
      <p>
        MediCore offers general health information and is{" "}
        <strong>not a substitute for professional medical advice</strong>. In an
        emergency, call your local emergency number immediately.
      </p>
    </div>
  );
}
