import { createFileRoute } from "@tanstack/react-router";
import { ChatWindow } from "@/components/chat/ChatWindow";

export const Route = createFileRoute("/_authenticated/chat/$conversationId")({
  component: ChatThread,
});

function ChatThread() {
  const { conversationId } = Route.useParams();
  return <ChatWindow key={conversationId} conversationId={conversationId} />;
}
