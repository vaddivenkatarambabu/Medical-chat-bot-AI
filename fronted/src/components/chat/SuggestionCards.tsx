import {
  Thermometer,
  HeartPulse,
  Brain,
  Pill,
  Bandage,
  Activity,
} from "lucide-react";
import { motion } from "framer-motion";

const suggestions = [
  {
    icon: Thermometer,
    title: "Persistent fever",
    prompt:
      "I've had a fever above 38.5°C for 3 days with body aches. What could it be and when should I see a doctor?",
  },
  {
    icon: HeartPulse,
    title: "Chest discomfort",
    prompt:
      "I'm experiencing mild chest tightness when climbing stairs. Should I be worried?",
  },
  {
    icon: Brain,
    title: "Headache patterns",
    prompt:
      "I've had recurring headaches behind my eyes every afternoon for a week. What might be causing this?",
  },
  {
    icon: Pill,
    title: "Medication question",
    prompt:
      "Can I take ibuprofen and acetaminophen together for a strong headache?",
  },
  {
    icon: Bandage,
    title: "Minor injury",
    prompt:
      "I cut my finger while cooking — it's bleeding but not deep. How do I care for it?",
  },
  {
    icon: Activity,
    title: "Lifestyle advice",
    prompt: "What's a sustainable way to lower my resting heart rate?",
  },
];

// Pre-compute transition objects once so they are not recreated on every render
const transitions = suggestions.map((_, i) => ({ delay: i * 0.04 }));

export function SuggestionCards({
  onPick,
}: {
  onPick: (text: string) => void;
}) {
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-3xl">
      {suggestions.map((s, i) => (
        <motion.button
          key={s.title}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={transitions[i]}
          onClick={() => onPick(s.prompt)}
          className="text-left glass-panel rounded-xl p-4 hover:border-primary/50 hover:shadow-[var(--shadow-elegant)] transition-all group"
        >
          <div className="flex items-center gap-2.5">
            <div className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary group-hover:bg-primary/20 transition">
              <s.icon className="h-4 w-4" />
            </div>
            <div className="text-sm font-medium">{s.title}</div>
          </div>
          <p className="mt-2 text-xs text-muted-foreground line-clamp-2">
            {s.prompt}
          </p>
        </motion.button>
      ))}
    </div>
  );
}
