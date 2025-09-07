// components/research/ChatProvider.tsx
"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

type Sender = "user" | "bot";
export interface Message {
  id: string;
  content: string;
  sender: Sender;
  timestamp: Date;
  sources?: { page: number; snippet: string; explanation: string }[];
}

export interface ChatSession {
  id: string;
  fileName: string;
  fileSize: number;
  messages: Message[];
  createdAt: Date;
  lastActive: Date;
}

export interface Formula {
  page: number;
  formula: string;
}

interface ChatContextValue {
  // state
  pdfFile: File | null;
  setPdfFile: (f: File | null) => void;
  pdfUrl: string | null;
  chatSessions: ChatSession[];
  activeChatId: string | null;
  setActiveChatId: (id: string | null) => void;
  mode: "Novice" | "Reviewer" | "Researcher";
  setMode: (m: "Novice" | "Reviewer" | "Researcher") => void;
  isLoading: boolean;
  isLoadingPdf: boolean;
  numPages: number | null;
  setNumPages: (n: number | null) => void;
  pageNumber: number;
  setPageNumber: (p: number) => void;

  // formulas / related papers
  formulas: Formula[];
  isFormulaSheetOpen: boolean;
  setIsFormulaSheetOpen: (v: boolean) => void;
  selectedFormula: Formula | null;
  isDialogOpen: boolean;
  setIsDialogOpen: (v: boolean) => void;
  explanation: string;
  isExplanationLoading: boolean;

  // derived
  activeChat: ChatSession | undefined;
  messagesEndRef: React.RefObject<HTMLDivElement>;

  // functions / handlers
  handleFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
  createNewChatSession: (file: File, fileName: string) => string;
  switchToChat: (sessionId: string) => Promise<void>;
  handleSendMessage: () => Promise<void>;
  inputMessage: string;
  setInputMessage: (s: string) => void;
  fetchContext: (page: number) => Promise<{ text?: string } | null>;
  handleFormulaClick: (formula: Formula) => Promise<void>;
  deleteChatSession: (sessionId: string) => Promise<void>;

  relatedPapers: { pdf: string; difference: string }[];
  isRelatedPaperLoading: boolean;
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // shared states
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);

  const [mode, setMode] = useState<"Novice" | "Reviewer" | "Researcher">("Novice");

  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingPdf, setIsLoadingPdf] = useState(false);

  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(1);

  // formulas
  const [formulas, setFormulas] = useState<Formula[]>([]);
  const [isFormulaSheetOpen, setIsFormulaSheetOpen] = useState(false);
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [explanation, setExplanation] = useState<string>("");
  const [isExplanationLoading, setIsExplanationLoading] = useState(false);

  const [inputMessage, setInputMessage] = useState("");
  const [relatedPapers, setRelatedPapers] = useState<{ pdf: string; difference: string }[]>([]);
  const [isRelatedPaperLoading, setIsRelatedPaperLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null) as React.RefObject<HTMLDivElement>;

  // Derived
  const activeChat = chatSessions.find((s) => s.id === activeChatId);

  // local effect to set pdfUrl from pdfFile (your original logic)
  useEffect(() => {
    if (pdfFile) {
      // you originally used file.name as url/identifier
      setPdfUrl(pdfFile.name);
    } else {
      setPdfUrl(null);
    }
  }, [pdfFile]);

  // load sessions from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("chatSessions");
    if (saved) {
      try {
        const parsed = JSON.parse(saved).map((session: any) => ({
          ...session,
          createdAt: new Date(session.createdAt),
          lastActive: new Date(session.lastActive),
          messages: session.messages.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) })),
        }));
        setChatSessions(parsed);
        if (parsed.length > 0) {
          setActiveChatId(parsed[0].id);
        }
      } catch (e) {
        console.error("Failed to parse saved sessions", e);
      }
    }
  }, []);

  // save to localStorage when sessions change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem("chatSessions", JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  // fetch related papers when pdfUrl changes
  useEffect(() => {
    if (!pdfUrl) {
      setRelatedPapers([]);
      setIsRelatedPaperLoading(false);
      return;
    }
    setIsRelatedPaperLoading(true);
    fetch(`${API_BASE_URL}/related_papers?pdf_filename=${encodeURIComponent(pdfUrl)}`)
      .then((r) => r.json())
      .then((data) => setRelatedPapers(data.related_papers || []))
      .catch((err) => {
        console.error("related papers error", err);
      })
      .finally(() => setIsRelatedPaperLoading(false));
  }, [pdfUrl]);

  // utilities
  const createNewChatSession = (file: File, fileName: string) => {
    const newSessionId = Date.now().toString();
    const newSession: ChatSession = {
      id: newSessionId,
      fileName,
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

  const updateActiveChatMessages = (newMessages: Message[]) => {
    setChatSessions((prev) => prev.map((s) => (s.id === activeChatId ? { ...s, messages: newMessages, lastActive: new Date() } : s)));
  };

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || file.type !== "application/pdf") return;
    setIsLoadingPdf(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const resp = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();

      // server returns { message: "uploaded", filename: "..." } in your original code
      if (data.message === "uploaded") {
        setPdfFile(file);
        const existing = chatSessions.find((s) => s.fileName === data.filename && s.fileSize === file.size);
        if (existing) {
          setActiveChatId(existing.id);
          await switchToChat(existing.id);
        } else {
          const newId = createNewChatSession(file, data.filename);
          setActiveChatId(newId);
        }
      } else {
        console.warn("Unexpected upload response", data);
      }
    } catch (err: any) {
      alert(`Error uploading PDF: ${err.message || err}`);
      console.error(err);
    } finally {
      setIsLoadingPdf(false);
    }
  }

  async function switchToChat(sessionId: string) {
    setActiveChatId(sessionId);
    setFormulas([]); // reset

    const session = chatSessions.find((s) => s.id === sessionId);
    if (!session) {
      setPdfFile(null);
      setFormulas([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/get_pdf/${encodeURIComponent(session.fileName)}`);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const file = new File([blob], session.fileName, { type: "application/pdf" });
      Object.defineProperty(file, "size", { value: session.fileSize });
      setPdfFile(file);

      // fetch formulas
      try {
        const formulaResp = await fetch(`${API_BASE_URL}/formulas?pdf_filename=${encodeURIComponent(session.fileName)}`);
        if (!formulaResp.ok) {
          throw new Error("Failed to fetch formulas");
        }
        const fd = await formulaResp.json();
        setFormulas(fd.formulas || []);
      } catch (e) {
        console.error("Error fetching formulas:", e);
      }
    } catch (err: any) {
      console.error("Error fetching PDF:", err);
      alert(`Error loading PDF: ${err.message || "Unable to fetch PDF from server"}`);
      setPdfFile(null);
      setFormulas([]);
    } finally {
      setIsLoading(false);
    }
  }

  async function fetchContext(page: number) {
    if (!activeChat) return null;
    try {
      const response = await fetch(`${API_BASE_URL}/context?pdf_filename=${encodeURIComponent(activeChat.fileName)}&page=${page}`);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      return data.page;
    } catch (err: any) {
      console.error("fetchContext error", err);
      alert(`Error fetching context: ${err.message || err}`);
      return null;
    }
  }

  async function handleSendMessage() {
    if (!inputMessage.trim() || !activeChatId) return;
    setIsLoading(true);
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    };
    const curSession = chatSessions.find((s) => s.id === activeChatId);
    const messages = curSession?.messages || [];
    updateActiveChatMessages([...messages, userMessage]);

    try {
      const params = new URLSearchParams({
        question: inputMessage,
        mode: mode,
        pdf_filename: curSession!.fileName,
        k: "3",
      });
      const response = await fetch(`${API_BASE_URL}/ask?${params}`, { method: "GET" });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        sender: "bot",
        timestamp: new Date(),
        sources: data.sources,
      };
      updateActiveChatMessages([...messages, userMessage, botMessage]);
    } catch (err: any) {
      const errMsg: Message = {
        id: (Date.now() + 1).toString(),
        content: `Error: ${err.message || "An unexpected error occurred"}`,
        sender: "bot",
        timestamp: new Date(),
      };
      updateActiveChatMessages([...messages, userMessage, errMsg]);
    } finally {
      setIsLoading(false);
      setInputMessage("");
    }
  }

  async function handleFormulaClick(formula: Formula) {
    if (!activeChat || !pdfFile) return;
    setSelectedFormula(formula);
    setIsDialogOpen(true);
    setIsExplanationLoading(true);
    setExplanation("");
    try {
      const params = new URLSearchParams({
        question: `Explain the variables and meaning of this formula from page ${formula.page}: ${formula.formula}`,
        mode: "normal",
        pdf_filename: activeChat.fileName,
        k: "5",
        isFormula: "true",
      });
      const response = await fetch(`${API_BASE_URL}/ask?${params}`);
      if (!response.ok) throw new Error("Failed to generate explanation");
      const data = await response.json();
      setExplanation(data.answer);
    } catch (e) {
      setExplanation("Error generating explanation. Please try again.");
      console.error(e);
    } finally {
      setIsExplanationLoading(false);
    }
  }

  async function deleteChatSession(sessionId: string) {
    const session = chatSessions.find((s) => s.id === sessionId);
    if (session) {
      setIsLoading(true);
      try {
        const resp = await fetch(`${API_BASE_URL}/delete_pdf/${encodeURIComponent(session.fileName)}`, { method: "DELETE" });
        if (!resp.ok) {
          const e = await resp.json().catch(() => ({}));
          throw new Error(e.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        // optionally toast here
        console.log(data.message);
      } catch (err: any) {
        console.error("Error deleting session:", err);
        alert(`Error deleting session: ${err.message || "Failed to delete session"}`);
      } finally {
        setIsLoading(false);
      }
    }

    setChatSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeChatId === sessionId) {
      const remaining = chatSessions.filter((s) => s.id !== sessionId);
      if (remaining.length > 0) {
        // switch to next one
        await switchToChat(remaining[0].id);
      } else {
        setActiveChatId(null);
        setPdfFile(null);
      }
    }
  }

  // Provide context values
  const ctx: ChatContextValue = {
    pdfFile,
    setPdfFile,
    pdfUrl,
    chatSessions,
    activeChatId,
    setActiveChatId,
    mode,
    setMode,
    isLoading,
    isLoadingPdf,
    numPages,
    setNumPages,
    pageNumber,
    setPageNumber,
    formulas,
    isFormulaSheetOpen,
    setIsFormulaSheetOpen,
    selectedFormula,
    isDialogOpen,
    setIsDialogOpen,
    explanation,
    isExplanationLoading,
    activeChat,
    messagesEndRef,
    handleFileUpload,
    createNewChatSession,
    switchToChat,
    handleSendMessage,
    inputMessage,
    setInputMessage,
    fetchContext,
    handleFormulaClick,
    deleteChatSession,
    relatedPapers,
    isRelatedPaperLoading,
  };

  return <ChatContext.Provider value={ctx}>{children}</ChatContext.Provider>;
};

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within ChatProvider");
  return ctx;
}
