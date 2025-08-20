"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Upload,
  Moon,
  Sun,
  Send,
  FileText,
  History,
  MessageSquare,
  Trash2,
  Plus,
} from "lucide-react";
import { useTheme } from "next-themes";

interface Message {
  id: string;
  content: string;
  sender: "user" | "bot";
  timestamp: Date;
  sources?: { page: number; snippet: string }[];
}

interface ChatSession {
  id: string;
  fileName: string;
  fileSize: number;
  messages: Message[];
  createdAt: Date;
  lastActive: Date;
}

const API_BASE_URL = "http://localhost:8000";

export default function ResearchPaperChat() {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { theme, setTheme } = useTheme();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeChat = chatSessions.find(
    (session) => session.id === activeChatId
  );
  const messages = activeChat?.messages || [];

  // Auto-scroll to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Load chat sessions from localStorage
  useEffect(() => {
    const savedSessions = localStorage.getItem("chatSessions");
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions).map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastActive: new Date(session.lastActive),
        messages: session.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })),
      }));
      setChatSessions(parsed);
      if (parsed.length > 0) {
        setActiveChatId(parsed[0].id);
      }
    }
  }, []);

  // Save chat sessions to localStorage
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem("chatSessions", JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  const createNewChatSession = (file: File, fileName: string): string => {
    const newSessionId = Date.now().toString();
    const newSession: ChatSession = {
      id: newSessionId,
      fileName: fileName,
      fileSize: file.size,
      createdAt: new Date(),
      lastActive: new Date(),
      messages: [
        {
          id: "1",
          content: `Hello! I'm ready to help you analyze "${fileName}". You can now ask me questions about this research paper.`,
          sender: "bot",
          timestamp: new Date(),
        },
      ],
    };
    setChatSessions((prev) => [newSession, ...prev]);
    return newSessionId;
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file || file.type !== "application/pdf") return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to upload PDF");
      }
      const data = await response.json();
      console.log("Upload response:", data.message);
      if (data.message === "uploaded") {
        setPdfFile(file);
        console.log("PDF uploaded successfully:", file.name);
        console.log("Pdf file:", pdfFile);
        const existingSession = chatSessions.find(
          (session) =>
            session.fileName === file.name && session.fileSize === file.size
        );
        if (existingSession) {
          setActiveChatId(existingSession.id);
        } else {
          const newSessionId = createNewChatSession(file, data.filename);
          setActiveChatId(newSessionId);
        }
      }
    } catch (error: any) {
      alert(`Error uploading PDF: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const updateActiveChat = (newMessages: Message[]) => {
    setChatSessions((prev) =>
      prev.map((session) =>
        session.id === activeChatId
          ? { ...session, messages: newMessages, lastActive: new Date() }
          : session
      )
    );
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !activeChatId || !activeChat) return;

    setIsLoading(true);
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    };
    const newMessages = [...messages, userMessage];
    updateActiveChat(newMessages);

    try {
      const params = new URLSearchParams({
        question: inputMessage,
        mode: "Novice",
        pdf_filename: activeChat.fileName,
        k: "3",
      });
      const response = await fetch(`${API_BASE_URL}/ask?${params}`, {
        method: "GET",
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error ${response.status}`);
      }
      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        sender: "bot",
        timestamp: new Date(),
        sources: data.sources,
      };
      updateActiveChat([...newMessages, botMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Error: ${error.message || "An unexpected error occurred"}`,
        sender: "bot",
        timestamp: new Date(),
      };
      updateActiveChat([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
      setInputMessage("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const switchToChat = (sessionId: string) => {
    setActiveChatId(sessionId);
    const session = chatSessions.find((s) => s.id === sessionId);
    if (session) {
      const mockFile = new File([""], session.fileName, {
        type: "application/pdf",
      });
      Object.defineProperty(mockFile, "size", { value: session.fileSize });
      setPdfFile(mockFile);
    }
  };

  const deleteChatSession = (sessionId: string) => {
    setChatSessions((prev) =>
      prev.filter((session) => session.id !== sessionId)
    );
    if (activeChatId === sessionId) {
      const remainingSessions = chatSessions.filter(
        (session) => session.id !== sessionId
      );
      if (remainingSessions.length > 0) {
        switchToChat(remainingSessions[0].id);
      } else {
        setActiveChatId(null);
        setPdfFile(null);
      }
    }
  };

  const createNewChat = () => {
    document.getElementById("pdf-upload")?.click();
  };

  const fetchContext = async (page: number) => {
    if (!activeChat) return;
    try {
      const response = await fetch(
        `${API_BASE_URL}/context?pdf_filename=${encodeURIComponent(
          activeChat.fileName
        )}&page=${page}`
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to fetch context");
      }
      const data = await response.json();
      return data.page;
    } catch (error: any) {
      alert(`Error fetching context: ${error.message}`);
      return null;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header Section */}
      <header className="border-b bg-card h-16">
        <div className="container mx-auto px-3 py-2">
          {" "}
          {/* Adjusted px-4 to px-3, py-3 to py-2 for tighter spacing */}
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
                {pdfFile && (
                  <span className="text-sm text-muted-foreground">
                    {pdfFile.name}
                  </span>
                )}
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
                      <MessageSquare className="h-5 w-5" />
                      Chat History
                    </SheetTitle>
                  </SheetHeader>
                  <div className="mt-6">
                    <div className="flex justify-between items-center mb-3">
                      {" "}
                      {/* Adjusted mb-4 to mb-3 */}
                      <p className="text-sm text-muted-foreground">
                        {chatSessions.length} conversation
                        {chatSessions.length !== 1 ? "s" : ""}
                      </p>
                      <Button
                        onClick={createNewChat}
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
                            {" "}
                            {/* Adjusted py-8 to py-6 */}
                            <MessageSquare className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />{" "}
                            {/* Adjusted mb-4 to mb-3 */}
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
                              className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                                session.id === activeChatId
                                  ? "ring-2 ring-primary"
                                  : ""
                              }`}
                              onClick={() => switchToChat(session.id)}
                            >
                              <CardContent className="p-3">
                                {" "}
                                {/* Adjusted p-4 to p-3 */}
                                <div className="flex items-start justify-between">
                                  <div className="flex-1 min-w-0">
                                    <h4 className="font-medium truncate mb-1">
                                      {session.fileName}
                                    </h4>
                                    <p className="text-sm text-muted-foreground mb-1">
                                      {" "}
                                      {/* Adjusted mb-2 to mb-1 */}
                                      {session.messages.length} message
                                      {session.messages.length !== 1 ? "s" : ""}
                                      {" • "}
                                      {(session.fileSize / 1024 / 1024).toFixed(
                                        1
                                      )}{" "}
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
                                    disabled={isLoading}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
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
                <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Section */}
      <div className="container mx-auto px-3 py-4">
        {" "}
        {/* Adjusted px-4 to px-3, py-6 to py-4 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-h-[calc(100vh-6rem)]">
          {/* PDF Card Section */}
          <Card className="flex flex-col max-h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Research Paper
                {activeChat && (
                  <Badge variant="outline" className="ml-auto">
                    Active Chat
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              <ScrollArea className="h-full pr-3">
                {" "}
                {/* Adjusted pr-4 to pr-3 */}
                {pdfFile ? (
                  <div className="space-y-3">
                    {" "}
                    {/* Adjusted space-y-4 to space-y-3 */}
                    <div className="p-3 bg-muted rounded-lg">
                      {" "}
                      {/* Adjusted p-4 to p-3 */}
                      <h3 className="font-semibold mb-2">
                        PDF Loaded: {pdfFile.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        File size: {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      {activeChat && (
                        <p className="text-sm text-muted-foreground mt-1">
                          Chat started:{" "}
                          {activeChat.createdAt.toLocaleDateString()}
                        </p>
                      )}
                    </div>
                    <div className="aspect-[3/4] bg-muted rounded-lg flex items-center justify-center">
                      <div className="text-center w-full h-full flex flex-col items-center justify-center">
                        {/* <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-muted-foreground">
                          PDF Preview would be displayed here
                        </p>
                        <p className="text-sm text-muted-foreground mt-2">
                          In a real implementation, you would integrate a PDF
                          viewer library
                        </p> */}

                        {pdfFile && (
                          <iframe
                            src={URL.createObjectURL(pdfFile)}
                            title="PDF Preview"
                            width="100%"
                            height="100%"
                          />
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Upload className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                      <h3 className="text-lg font-semibold mb-2">
                        No PDF Loaded
                      </h3>
                      <p className="text-muted-foreground mb-3">
                        {" "}
                        {/* Adjusted mb-4 to mb-3 */}
                        Upload a research paper to get started or select from
                        chat history
                      </p>
                      <div className="flex gap-2 justify-center">
                        <Button
                          onClick={() =>
                            document.getElementById("pdf-upload")?.click()
                          }
                          className="gap-2"
                          disabled={isLoading}
                        >
                          <Upload className="h-4 w-4" />
                          Choose PDF File
                        </Button>
                        {chatSessions.length > 0 && (
                          <Sheet>
                            <SheetTrigger asChild>
                              <Button
                                variant="outline"
                                className="gap-2"
                                disabled={isLoading}
                              >
                                <History className="h-4 w-4" />
                                View History
                              </Button>
                            </SheetTrigger>
                            <SheetContent className="w-[400px] sm:w-[540px]">
                              <SheetHeader>
                                <SheetTitle>Select a Previous Chat</SheetTitle>
                              </SheetHeader>
                              <div className="mt-6">
                                <ScrollArea className="h-[calc(100vh-200px)]">
                                  <div className="space-y-3">
                                    {chatSessions.map((session) => (
                                      <Card
                                        key={session.id}
                                        className="cursor-pointer transition-colors hover:bg-muted/50"
                                        onClick={() => switchToChat(session.id)}
                                      >
                                        <CardContent className="p-4">
                                          <h4 className="font-medium mb-1">
                                            {session.fileName}
                                          </h4>
                                          <p className="text-sm text-muted-foreground">
                                            {session.messages.length} messages •{" "}
                                            {session.lastActive.toLocaleDateString()}
                                          </p>
                                        </CardContent>
                                      </Card>
                                    ))}
                                  </div>
                                </ScrollArea>
                              </div>
                            </SheetContent>
                          </Sheet>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Chatting Card Section */}
          <Card className="flex flex-col max-h-[calc(100vh-6rem)]">
            <CardHeader className="h-16">
              <CardTitle className="flex items-center justify-between">
                <span>AI Assistant</span>
                {activeChat && (
                  <Badge variant="secondary" className="text-xs">
                    {messages.length} messages
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col overflow-hidden">
              <ScrollArea className="h-[calc(100%-4rem)] pr-3">
                {" "}
                {/* Adjusted pr-4 to pr-3 */}
                <div className="space-y-3 p-2">
                  {" "}
                  {/* Adjusted space-y-4 to space-y-3, kept p-2 */}
                  {messages.length === 0 ? (
                    <div className="text-center py-6">
                      {" "}
                      {/* Adjusted py-8 to py-6 */}
                      <MessageSquare className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />{" "}
                      {/* Adjusted mb-4 to mb-3 */}
                      <p className="text-muted-foreground">No active chat</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Upload a PDF or select from chat history to start
                      </p>
                    </div>
                  ) : (
                    messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${
                          message.sender === "user"
                            ? "justify-end"
                            : "justify-start"
                        }`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg px-3 py-2 break-words ${
                            message.sender === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <p className="text-sm whitespace-pre-wrap">
                            {message.content}
                          </p>
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-2">
                              <p className="text-xs font-semibold">Sources:</p>
                              {message.sources.map((source, index) => (
                                <p
                                  key={index}
                                  className="text-xs text-muted-foreground cursor-pointer hover:underline whitespace-pre-wrap"
                                  onClick={() =>
                                    fetchContext(source.page).then(
                                      (context) => {
                                        if (context) {
                                          alert(
                                            `Page ${
                                              source.page
                                            }: ${context.text.substring(
                                              0,
                                              500
                                            )}...`
                                          );
                                        }
                                      }
                                    )
                                  }
                                >
                                  Page {source.page}:{" "}
                                  {source.snippet.substring(0, 100)}...
                                </p>
                              ))}
                            </div>
                          )}
                          <p className="text-xs opacity-70 mt-1">
                            {message.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
              <Separator className="my-3" /> {/* Adjusted my-4 to my-3 */}
              <div className="flex gap-2 p-1">
                {" "}
                {/* Adjusted p-2 to p-1 */}
                <Input
                  placeholder={
                    activeChat
                      ? "Ask questions about the research paper..."
                      : "Upload a PDF to start chatting..."
                  }
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="flex-1"
                  disabled={!activeChat || isLoading}
                />
                <Button
                  onClick={handleSendMessage}
                  size="icon"
                  disabled={!activeChat || !inputMessage.trim() || isLoading}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
