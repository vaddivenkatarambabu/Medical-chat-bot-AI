import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  ShieldCheck,
  Stethoscope,
  Sparkles,
  MessageSquareText,
  Activity,
  HeartPulse,
  Lock,
} from "lucide-react";
import hero from "@/assets/hero-medical.jpg";
import { Logo } from "@/components/brand/Logo";
import { Blobs } from "@/components/landing/Blobs";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      {
        title: "MediCore — Trusted AI Medical Assistant",
      },
      {
        name: "description",
        content:
          "Premium AI healthcare assistant for symptom guidance, triage, and wellness — designed for clarity, empathy, and trust.",
      },
      {
        property: "og:title",
        content: "MediCore — Trusted AI Medical Assistant",
      },
      {
        property: "og:description",
        content:
          "Premium AI healthcare assistant for symptom guidance, triage, and wellness.",
      },
    ],
  }),
  component: Landing,
});

const features = [
  {
    icon: Brain,
    title: "Clinical-grade reasoning",
    body: "Trained guidance that asks the right follow-ups — onset, duration, severity, context.",
  },
  {
    icon: ShieldCheck,
    title: "Private & encrypted",
    body: "Your conversations are stored privately and encrypted in transit. Always yours.",
  },
  {
    icon: Stethoscope,
    title: "Triage-first design",
    body: "Clear next steps, emergency flags, and recommendations from urgent care to home care.",
  },
  {
    icon: Sparkles,
    title: "Empathetic responses",
    body: "Calm, structured replies in plain language — not jargon.",
  },
  {
    icon: Activity,
    title: "Symptom tracking",
    body: "Save consultations and revisit them over time.",
  },
  {
    icon: Lock,
    title: "Disclaimer-first",
    body: "Educational tool — never a replacement for licensed clinicians.",
  },
];

const testimonials = [
  {
    name: "Dr. Amara Okafor",
    role: "Internal Medicine",
    quote:
      "The structured triage prompts mirror how I take a history. Impressive baseline.",
  },
  {
    name: "Liam Carter",
    role: "Product Manager",
    quote:
      "I used MediCore at 2 a.m. for my kid's fever. It calmed me down and told me when to go in.",
  },
  {
    name: "Priya Shah",
    role: "RN, ER",
    quote:
      "Clear, conservative, and it never pretends to diagnose. Exactly what a patient-facing AI should be.",
  },
];

const faqs = [
  {
    q: "Is MediCore a replacement for a doctor?",
    a: "No. MediCore provides general health education and triage guidance. Always consult a licensed clinician for diagnosis and treatment.",
  },
  {
    q: "How is my data handled?",
    a: "Conversations are stored privately to your account and protected by row-level security. You can delete any consultation at any time.",
  },
  {
    q: "Does MediCore handle emergencies?",
    a: "If you describe an emergency, MediCore will direct you to call your local emergency number immediately.",
  },
  {
    q: "Which AI powers MediCore?",
    a: "A state-of-the-art reasoning model fine-tuned with a medical safety prompt and triage workflow.",
  },
];

export default function Landing() {
  return (
    <div className="relative min-h-dvh overflow-hidden bg-background">
      <Blobs />

      <div
        className="absolute inset-0"
        style={{
          background: "var(--gradient-hero)",
        }}
        aria-hidden
      />

      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <Logo />

        <nav className="hidden items-center gap-8 text-sm text-muted-foreground md:flex">
          <a href="#features" className="transition hover:text-foreground">
            Features
          </a>

          <a href="#testimonials" className="transition hover:text-foreground">
            Testimonials
          </a>

          <a href="#faq" className="transition hover:text-foreground">
            FAQ
          </a>
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle />

          <Button asChild variant="ghost">
            <Link to="/sign-in">Sign in</Link>
          </Button>

          <Button asChild className="shadow-[var(--shadow-elegant)]">
            <Link to="/chat">
              Launch app
              <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </header>

      <section className="relative z-10 mx-auto grid max-w-7xl items-center gap-12 px-6 pb-24 pt-12 lg:grid-cols-2">
        <motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 0.6,
          }}
        >
          <div className="inline-flex items-center gap-2 rounded-full border bg-card/50 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            Powered by AI · Medical Safety First
          </div>

          <h1 className="mt-6 font-display text-5xl font-bold leading-[1.05] tracking-tight md:text-6xl">
            Your <span className="gradient-text">AI medical</span> companion,
            always on call.
          </h1>

          <p className="mt-6 max-w-xl text-lg text-muted-foreground">
            MediCore combines clinical-grade reasoning with empathetic
            conversation. Get structured symptom guidance, triage support, and
            clear next steps — 24/7.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild size="lg" className="shadow-[var(--shadow-glow)]">
              <Link to="/chat">
                Start a consultation
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>

            <Button asChild size="lg" variant="outline">
              <Link to="/sign-up">Create free account</Link>
            </Button>
          </div>

          <div className="mt-10 flex items-center gap-6 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-primary" />
              End-to-end encrypted
            </div>

            <div className="flex items-center gap-2">
              <HeartPulse className="h-4 w-4 text-primary" />
              Clinician-reviewed prompts
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{
            opacity: 0,
            scale: 0.95,
          }}
          animate={{
            opacity: 1,
            scale: 1,
          }}
          transition={{
            duration: 0.7,
            delay: 0.1,
          }}
          className="relative"
        >
          <div className="glass-panel rounded-3xl p-2 shadow-[var(--shadow-elegant)]">
            <img
              src={hero}
              alt="MediCore AI medical assistant"
              width={1536}
              height={1024}
              loading="eager"
              decoding="async"
              className="h-auto w-full rounded-2xl"
            />
          </div>
        </motion.div>
      </section>

      <section
        id="features"
        className="relative z-10 mx-auto max-w-7xl px-6 py-24"
      >
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-3xl font-bold md:text-4xl">
            Built for trust. Designed for clarity.
          </h2>

          <p className="mt-4 text-muted-foreground">
            Every interaction is structured around safety, transparency, and
            your wellbeing.
          </p>
        </div>

        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{
                opacity: 0,
                y: 20,
              }}
              whileInView={{
                opacity: 1,
                y: 0,
              }}
              viewport={{
                once: true,
              }}
              transition={{
                delay: index * 0.05,
              }}
              className="glass-panel rounded-2xl p-6 transition-shadow hover:shadow-[var(--shadow-elegant)]"
            >
              <div className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <feature.icon className="h-5 w-5" />
              </div>

              <h3 className="mt-4 font-semibold">{feature.title}</h3>

              <p className="mt-2 text-sm text-muted-foreground">
                {feature.body}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      <section
        id="testimonials"
        className="relative z-10 mx-auto max-w-7xl px-6 py-24"
      >
        <h2 className="text-center font-display text-3xl font-bold md:text-4xl">
          Trusted by clinicians and patients
        </h2>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {testimonials.map((testimonial) => (
            <div key={testimonial.name} className="glass-panel rounded-2xl p-6">
              <MessageSquareText className="h-5 w-5 text-primary" />

              <p className="mt-4 text-sm leading-relaxed">
                "{testimonial.quote}"
              </p>

              <div className="mt-6">
                <div className="text-sm font-semibold">{testimonial.name}</div>

                <div className="text-xs text-muted-foreground">
                  {testimonial.role}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section id="faq" className="relative z-10 mx-auto max-w-3xl px-6 py-24">
        <h2 className="text-center font-display text-3xl font-bold md:text-4xl">
          Frequently asked
        </h2>

        <Accordion
          type="single"
          collapsible
          className="glass-panel mt-10 rounded-2xl px-6"
        >
          {faqs.map((faq, index) => (
            <AccordionItem key={faq.q} value={`item-${index}`}>
              <AccordionTrigger>{faq.q}</AccordionTrigger>

              <AccordionContent className="text-muted-foreground">
                {faq.a}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </section>

      <section className="relative z-10 mx-auto max-w-5xl px-6 pb-24">
        <div className="glass-panel rounded-3xl p-10 text-center shadow-[var(--shadow-elegant)]">
          <h2 className="font-display text-3xl font-bold md:text-4xl">
            Get answers when it matters most.
          </h2>

          <p className="mt-4 text-muted-foreground">
            Start a free consultation in seconds. No credit card required.
          </p>

          <Button asChild size="lg" className="mt-8">
            <Link to="/chat">
              Launch MediCore
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>

      <footer className="relative z-10 mt-12 border-t">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-8 text-sm text-muted-foreground md:flex-row">
          <div className="flex items-center gap-3">
            <Logo size={24} />
            <span>
              © {new Date().getFullYear()} MediCore. Not a medical device.
            </span>
          </div>

          <div className="flex items-center gap-6">
            <a href="#features" className="hover:text-foreground">
              Features
            </a>

            <a href="#faq" className="hover:text-foreground">
              FAQ
            </a>

            <Link to="/sign-in" className="hover:text-foreground">
              Sign in
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
