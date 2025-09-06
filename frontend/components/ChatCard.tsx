// components/research/ChatCard.tsx
"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { Send, MessageSquare } from "lucide-react";
import { useChat } from "./ChatProvider";

export const ChatCard: React.FC = () => {
  const {
    chatSessions,
    activeChat,
    messagesEndRef,
    inputMessage,
    setInputMessage,
    handleSendMessage,
    fetchContext,
    relatedPapers,
    isRelatedPaperLoading,
    isLoading,
  } = useChat();

  // local UI state
  const [openPopover, setOpenPopover] = useState<{ msgId: string; idx: number } | null>(null);
  const [isContextDialogOpen, setIsContextDialogOpen] = useState(false);
  const [contextText, setContextText] = useState<string>("");
  const [openRelatedPopover, setOpenRelatedPopover] = useState<number | null>(null);

  // auto scroll
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeChat?.messages?.length, messagesEndRef]);

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const onSourceClick = async (source: { page: number; snippet: string; explanation: string }) => {
    const ctx = await fetchContext(source.page);
    if (ctx) {
      setContextText(ctx.text || "");
      setIsContextDialogOpen(true);
    } else {
      setContextText("No context found.");
      setIsContextDialogOpen(true);
    }
  };

  const messages = activeChat?.messages || [];

  return (
    <Card className="flex flex-col max-h-[calc(100vh-6rem)]">
      <CardHeader className="h-16">
        <CardTitle className="flex items-center justify-between">
          <span>AI Assistant</span>
          {activeChat && <Badge variant="secondary" className="text-xs">{messages.length} messages</Badge>}
        </CardTitle>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col overflow-hidden">
        <ScrollArea className="h-[calc(100%-4rem)] pr-3">
          <div className="space-y-3 p-2">
            {messages.length === 0 ? (
              <div className="text-center py-6">
                <MessageSquare className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                <p className="text-muted-foreground">No active chat</p>
                <p className="text-sm text-muted-foreground mt-1">Upload a PDF or select from chat history to start</p>
              </div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[80%] rounded-lg px-3 py-2 break-words ${message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-semibold">Sources:</p>
                        {message.sources.map((source, index) => (
                          <Popover key={index} open={openPopover?.msgId === message.id && openPopover?.idx === index} onOpenChange={(open) => { if (!open) setOpenPopover(null); }}>
                            <PopoverTrigger asChild>
                              <p
                                className="text-xs text-muted-foreground cursor-pointer hover:underline whitespace-pre-wrap"
                                onMouseEnter={() => setOpenPopover({ msgId: message.id, idx: index })}
                                onMouseLeave={() => setOpenPopover(null)}
                                onClick={() => onSourceClick(source)}
                              >
                                Page {source.page}: {source.snippet.substring(0, 100)}...
                              </p>
                            </PopoverTrigger>
                            <PopoverContent side="top" align="start" className="max-w-lg text-xs">
                              <div className="font-semibold mb-1">Why this?</div>
                              <div className="whitespace-pre-wrap">{source.explanation}</div>
                            </PopoverContent>
                          </Popover>
                        ))}
                      </div>
                    )}

                    <p className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</p>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <Separator className="my-3" />

        <div className="flex gap-2 p-1">
          <Input placeholder={activeChat ? "Ask questions about the research paper..." : "Upload a PDF to start chatting..."} value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} onKeyPress={handleKeyPress} className="flex-1" disabled={!activeChat || isLoading} />
          <Button onClick={handleSendMessage} size="icon" disabled={!activeChat || !inputMessage.trim() || isLoading}><Send className="h-4 w-4" /></Button>
        </div>
      </CardContent>

      {/** Related papers as footer */}
      <CardFooter>
        <div>
          <p className="text-xs font-semibold">Related Papers:</p>
          {isRelatedPaperLoading ? <p className="text-xs text-muted-foreground">Loading related papers...</p> : relatedPapers.length > 0 ? (
            <div className="mt-2">
              <div className="flex flex-wrap flex-col">
                {relatedPapers.map((paper, idx) => (
                  <a key={idx} href="#" className="text-xs text-blue-600 underline cursor-pointer mb-4" onClick={(e) => { e.preventDefault(); /* clicking behavior can be implemented: create session + fetch pdf */ }}>
                    {paper.pdf}
                  </a>
                ))}
              </div>
            </div>
          ) : <p className="text-xs text-muted-foreground">No related papers found.</p>}
        </div>
      </CardFooter>

      <Dialog open={isContextDialogOpen} onOpenChange={setIsContextDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <VisuallyHidden>
              <DialogTitle>Context</DialogTitle>
            </VisuallyHidden>
          </DialogHeader>
          <div className="whitespace-pre-wrap text-sm">{contextText || "No context found."}</div>
        </DialogContent>
      </Dialog>
    </Card>
  );
};
