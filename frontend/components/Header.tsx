// components/research/Header.tsx
"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Trash2, History, Plus, Upload, Sun, Moon } from "lucide-react";
import { useTheme } from "next-themes";
import { useChat } from "./ChatProvider";

export const Header: React.FC = () => {
  const {
    handleFileUpload,
    chatSessions,
    switchToChat,
    deleteChatSession,
    setMode,
    mode,
    isLoading,
  } = useChat();

  const { theme, setTheme } = useTheme();

  return (
    <header className="border-b bg-card h-16">
      <div className="container mx-auto px-3 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold">Research Paper Assistant</h1>
            <div className="flex items-center gap-2">
              <Input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="hidden"
                id="pdf-upload"
                disabled={isLoading}
              />
              <Button
                onClick={() => document.getElementById("pdf-upload")?.click()}
                variant="outline"
                className="gap-2"
                disabled={isLoading}
              >
                <Upload className="h-4 w-4" />
                Import PDF
              </Button>

              <Select
                value={mode}
                onValueChange={(v: any) => setMode(v)}
                disabled={isLoading}
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Select mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Novice">Novice</SelectItem>
                  <SelectItem value="Reviewer">Reviewer</SelectItem>
                  <SelectItem value="Researcher">Researcher</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="outline"
                  className="gap-2"
                  disabled={isLoading}
                >
                  <History className="h-4 w-4" />
                  Chat History
                  {chatSessions.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {chatSessions.length}
                    </Badge>
                  )}
                </Button>
              </SheetTrigger>

              <SheetContent className="w-[400px] sm:w-[540px]">
                <SheetHeader>
                  <SheetTitle className="flex items-center gap-2">
                    Chat History
                  </SheetTitle>
                </SheetHeader>

                <div className="mt-6">
                  <div className="flex justify-between items-center mb-3">
                    <p className="text-sm text-muted-foreground">
                      {chatSessions.length} conversation
                      {chatSessions.length !== 1 ? "s" : ""}
                    </p>
                    <Button
                      onClick={() =>
                        document.getElementById("pdf-upload")?.click()
                      }
                      size="sm"
                      className="gap-2"
                      disabled={isLoading}
                    >
                      <Plus className="h-4 w-4" />
                      New Chat
                    </Button>
                  </div>

                  <ScrollArea className="h-[calc(100vh-200px)]">
                    <div className="space-y-3">
                      {chatSessions.length === 0 ? (
                        <div className="text-center py-6">
                          <p className="text-muted-foreground">
                            No chat history yet
                          </p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Upload a PDF to start your first conversation
                          </p>
                        </div>
                      ) : (
                        chatSessions.map((session) => (
                          <Card
                            key={session.id}
                            className="cursor-pointer transition-colors hover:bg-muted/50"
                            onClick={() => switchToChat(session.id)}
                          >
                            <CardContent className="p-3 flex items-center justify-between">
                              <div>
                                <h4 className="font-medium truncate mb-1">
                                  {session.fileName}
                                </h4>
                                <p className="text-sm text-muted-foreground mb-1">
                                  {session.messages.length} message
                                  {session.messages.length !== 1
                                    ? "s"
                                    : ""} •{" "}
                                  {(session.fileSize / 1024 / 1024).toFixed(1)}{" "}
                                  MB
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  Last active:{" "}
                                  {session.lastActive.toLocaleDateString()}
                                </p>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  deleteChatSession(session.id);
                                }}
                                className="text-muted-foreground hover:text-destructive"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </CardContent>
                          </Card>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </SheetContent>
            </Sheet>

            <Button
              variant="outline"
              size="icon"
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}
              disabled={isLoading}
            >
              <span className="sr-only">Toggle theme</span>
              {/* icons omitted for brevity in header - you can add them similarly */}
              {theme === "light" ? (
                <Moon className="h-5 w-5" />
              ) : (
                <Sun className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};
