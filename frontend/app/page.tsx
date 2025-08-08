"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Upload, Moon, Sun, Send, FileText, History, MessageSquare, Trash2, Plus } from 'lucide-react'
import { useTheme } from "next-themes"

interface Message {
  id: string
  content: string
  sender: "user" | "bot"
  timestamp: Date
}

interface ChatSession {
  id: string
  fileName: string
  fileSize: number
  messages: Message[]
  createdAt: Date
  lastActive: Date
}

export default function ResearchPaperChat() {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [inputMessage, setInputMessage] = useState("")
  const { theme, setTheme } = useTheme()
  

  // Get current active chat session
  const activeChat = chatSessions.find(session => session.id === activeChatId)
  const messages = activeChat?.messages || []

  // Load chat sessions from localStorage on mount
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions')
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions).map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastActive: new Date(session.lastActive),
        messages: session.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
      }))
      setChatSessions(parsed)
      if (parsed.length > 0) {
        setActiveChatId(parsed[0].id)
      }
    }
  }, [])

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions))
    }
  }, [chatSessions])

  const createNewChatSession = (file: File): string => {
    const newSessionId = Date.now().toString()
    const newSession: ChatSession = {
      id: newSessionId,
      fileName: file.name,
      fileSize: file.size,
      createdAt: new Date(),
      lastActive: new Date(),
      messages: [
        {
          id: "1",
          content: `Hello! I'm ready to help you analyze "${file.name}". You can now ask me questions about this research paper.`,
          sender: "bot",
          timestamp: new Date()
        }
      ]
    }

    setChatSessions(prev => [newSession, ...prev])
    return newSessionId
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type === "application/pdf") {
      setPdfFile(file)
      
      // Check if we already have a session for this file
      const existingSession = chatSessions.find(session => 
        session.fileName === file.name && session.fileSize === file.size
      )

      if (existingSession) {
        setActiveChatId(existingSession.id)
      } else {
        const newSessionId = createNewChatSession(file)
        setActiveChatId(newSessionId)
      }
    }
  }

  const updateActiveChat = (newMessages: Message[]) => {
    setChatSessions(prev => prev.map(session => 
      session.id === activeChatId 
        ? { ...session, messages: newMessages, lastActive: new Date() }
        : session
    ))
  }

  const handleSendMessage = () => {
    if (!inputMessage.trim() || !activeChatId) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date()
    }

    const newMessages = [...messages, userMessage]
    updateActiveChat(newMessages)

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: activeChat 
          ? `I understand you're asking about "${inputMessage}". Based on the uploaded paper "${activeChat.fileName}", I can help analyze the content. This is a simulated response - in a real implementation, I would process the PDF content to provide specific insights.`
          : "Please upload a PDF file first so I can help you analyze the research paper.",
        sender: "bot",
        timestamp: new Date()
      }
      updateActiveChat([...newMessages, botMessage])
    }, 1000)

    setInputMessage("")
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const switchToChat = (sessionId: string) => {
    setActiveChatId(sessionId)
    const session = chatSessions.find(s => s.id === sessionId)
    if (session) {
      // Create a mock file object for display purposes
      const mockFile = new File([""], session.fileName, { type: "application/pdf" })
      Object.defineProperty(mockFile, 'size', { value: session.fileSize })
      setPdfFile(mockFile)
    }
  }

  const deleteChatSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(session => session.id !== sessionId))
    if (activeChatId === sessionId) {
      const remainingSessions = chatSessions.filter(session => session.id !== sessionId)
      if (remainingSessions.length > 0) {
        switchToChat(remainingSessions[0].id)
      } else {
        setActiveChatId(null)
        setPdfFile(null)
      }
    }
  }

  const createNewChat = () => {
    document.getElementById("pdf-upload")?.click()
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
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
                />
                <Button
                  onClick={() => document.getElementById("pdf-upload")?.click()}
                  variant="outline"
                  className="gap-2"
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
              {/* Chat History Sheet */}
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="outline" className="gap-2">
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
                    <div className="flex justify-between items-center mb-4">
                      <p className="text-sm text-muted-foreground">
                        {chatSessions.length} conversation{chatSessions.length !== 1 ? 's' : ''}
                      </p>
                      <Button onClick={createNewChat} size="sm" className="gap-2">
                        <Plus className="h-4 w-4" />
                        New Chat
                      </Button>
                    </div>
                    <ScrollArea className="h-[calc(100vh-200px)]">
                      <div className="space-y-3">
                        {chatSessions.length === 0 ? (
                          <div className="text-center py-8">
                            <MessageSquare className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                            <p className="text-muted-foreground">No chat history yet</p>
                            <p className="text-sm text-muted-foreground mt-1">
                              Upload a PDF to start your first conversation
                            </p>
                          </div>
                        ) : (
                          chatSessions.map((session) => (
                            <Card 
                              key={session.id} 
                              className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                                session.id === activeChatId ? 'ring-2 ring-primary' : ''
                              }`}
                              onClick={() => switchToChat(session.id)}
                            >
                              <CardContent className="p-4">
                                <div className="flex items-start justify-between">
                                  <div className="flex-1 min-w-0">
                                    <h4 className="font-medium truncate mb-1">
                                      {session.fileName}
                                    </h4>
                                    <p className="text-sm text-muted-foreground mb-2">
                                      {session.messages.length} message{session.messages.length !== 1 ? 's' : ''}
                                      {' • '}
                                      {(session.fileSize / 1024 / 1024).toFixed(1)} MB
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                      Last active: {session.lastActive.toLocaleDateString()}
                                    </p>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      deleteChatSession(session.id)
                                    }}
                                    className="text-muted-foreground hover:text-destructive"
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
              >
                <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-140px)]">
          {/* Left Column - Research Paper Viewer */}
          <Card className="flex flex-col">
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
            <CardContent className="flex-1">
              <ScrollArea className="h-full">
                {pdfFile ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h3 className="font-semibold mb-2">PDF Loaded: {pdfFile.name}</h3>
                      <p className="text-sm text-muted-foreground">
                        File size: {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      {activeChat && (
                        <p className="text-sm text-muted-foreground mt-1">
                          Chat started: {activeChat.createdAt.toLocaleDateString()}
                        </p>
                      )}
                    </div>
                    <div className="aspect-[3/4] bg-muted rounded-lg flex items-center justify-center">
                      <div className="text-center">
                        <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-muted-foreground">
                          PDF Preview would be displayed here
                        </p>
                        <p className="text-sm text-muted-foreground mt-2">
                          In a real implementation, you would integrate a PDF viewer library
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Upload className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                      <h3 className="text-lg font-semibold mb-2">No PDF Loaded</h3>
                      <p className="text-muted-foreground mb-4">
                        Upload a research paper to get started or select from chat history
                      </p>
                      <div className="flex gap-2 justify-center">
                        <Button
                          onClick={() => document.getElementById("pdf-upload")?.click()}
                          className="gap-2"
                        >
                          <Upload className="h-4 w-4" />
                          Choose PDF File
                        </Button>
                        {chatSessions.length > 0 && (
                          <Sheet>
                            <SheetTrigger asChild>
                              <Button variant="outline" className="gap-2">
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
                                          <h4 className="font-medium mb-1">{session.fileName}</h4>
                                          <p className="text-sm text-muted-foreground">
                                            {session.messages.length} messages • {session.lastActive.toLocaleDateString()}
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

          {/* Right Column - Chatbot */}
          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>AI Assistant</span>
                {activeChat && (
                  <Badge variant="secondary" className="text-xs">
                    {messages.length} messages
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              {/* Messages */}
              <ScrollArea className="flex-1 mb-4">
                <div className="space-y-4">
                  {messages.length === 0 ? (
                    <div className="text-center py-8">
                      <MessageSquare className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
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
                          message.sender === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg px-4 py-2 ${
                            message.sender === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <p className="text-sm">{message.content}</p>
                          <p className="text-xs opacity-70 mt-1">
                            {message.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>

              <Separator className="mb-4" />

              {/* Input */}
              <div className="flex gap-2">
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
                  disabled={!activeChat}
                />
                <Button 
                  onClick={handleSendMessage} 
                  size="icon"
                  disabled={!activeChat || !inputMessage.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
