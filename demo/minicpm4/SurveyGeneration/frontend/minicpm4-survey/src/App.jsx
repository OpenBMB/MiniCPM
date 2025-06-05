import React, { useState, useEffect, useMemo, useRef  } from 'react';
import './App.css';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

// 自定义 hook：防抖
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  React.useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

function MarkdownEditor({ value }) {
  const containerRef = useRef(null);

  const htmlContent = marked(value || '');
  const sanitizedHtml = DOMPurify.sanitize(htmlContent);

  const [userScrolled, setUserScrolled] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (container && !userScrolled) {
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
    }
  }, [value, userScrolled]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      const handleScroll = () => {
        const atBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 10;
        setUserScrolled(!atBottom);
      };
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, []);

   // 复制 Markdown 内容
  const handleCopy = () => {
    navigator.clipboard.writeText(value || '')
      .then(() => alert('Markdown 已复制到剪贴板'))
      .catch(err => console.error('复制失败:', err));
  };

  // 下载 Markdown 文件
  const handleDownload = () => {
    const blob = new Blob([value || ''], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'document.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="markdown-editor">
      {/* <div className="markdown-toolbar">
        <button className="neon-button" onClick={handleCopy}>复制 Markdown</button>
        <button className="neon-button" onClick={handleDownload}>下载 Markdown</button>
        </div> */}
      <div
        ref={containerRef}
        className="markdown-preview"
        dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
      />
    </div>
  );
}
function SendRequestToBackend() {
  const [inputValue, setInputValue] = useState('');

  const handleSendRequest = async () => {
    try {
      const response = await fetch('http://localhost:8001/generate_survey', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Failed to send request');
      }

      const data = await response.json();
      console.log('Response from backend:', data);
    } catch (error) {
      console.error('Error sending request:', error);
    }
  };

  return (
    <div className="request-panel" style={{ flexDirection: 'column', alignItems: 'center' }}>
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="neon-input"
        placeholder="Enter text to send"
        rows={3}
      />
      <button onClick={handleSendRequest} className="neon-button">
        Go!
      </button>
    </div>
  );
}


function App() {
  const [inputs, setInputs] = useState({
    query: { title: 'Query', displayText: '', targetText: '', isTyping: false },
    nowUpdate: { title: 'Now Update', displayText: '', targetText: '', isTyping: false },
    nextUpdate: { title: 'Next Update', displayText: '', targetText: '', isTyping: false },
    searchKeywords: { title: 'Search Keywords', displayText: '', targetText: '', isTyping: false },
    papers: { title: 'Papers', displayText: '', targetText: '', isTyping: false },
  });

  const [markdownContent, setMarkdownContent] = useState('');

  const inputKeyMap = {
    query: inputs.query,
    nowUpdate: inputs.nowUpdate,
    nextUpdate: inputs.nextUpdate,
    searchKeywords: inputs.searchKeywords,
    papers: inputs.papers,
    markdown: markdownContent
  };

  const updateInputsFromPostData = (postData) => {
    let newMarkdownContent = markdownContent;
    
    Object.entries(postData).forEach(([key, value]) => {
      if (key in inputKeyMap) {
      if (key === 'markdown') {
        if (markdownContent !== value) {
          newMarkdownContent = value;
          setMarkdownContent(newMarkdownContent);
        }
      } else if (inputKeyMap[key] && inputKeyMap[key].targetText !== value) {
        const updatedInput = {
          ...inputKeyMap[key],
          targetText: value,
          isTyping: true,
        };
        setInputs((prevInputs) => ({
          ...prevInputs,
          [key]: updatedInput,
        }));

        // startTypingAnimationForTextbox(value, (newText) => {
        //   setInputs((prevInputs) => ({
        // ...prevInputs,
        // [key]: {
        //   ...prevInputs[key],
        //   displayText: newText,
        // },
        //   }));
        // });
      }
    }
    });

  
  };

  // const startTypingAnimationForTextbox = (text, setText) => {
  //   setText('');
  //   let charIndex = 0;
  //   const timer = setInterval(() => {
  //     if (charIndex < text.length) {
  //       setText((prev) => prev + text[charIndex]);
  //       charIndex++;
  //     } else {
  //       clearInterval(timer);
  //     }
  //   }, 50); // Reduced interval for faster typing animation
  // };


  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8001/ws');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateInputsFromPostData(data);
        console.log('Received data:', data);
      } catch (e) {
        console.error('Invalid WebSocket message:', e);
      }
    };
    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
    // return () => ws.close();
  }, []);

  const leftInputs = [inputs.nowUpdate, inputs.nextUpdate,inputs.searchKeywords];
  const rightInputs = [inputs.papers];

  return (
    <div className="cyber-container">
      <div className="tech-panel left-panel">
        {leftInputs.map((input, index) => (
          <div key={`left-${index}`} className="input-wrapper">
            <h3 className="input-title" style={{ fontSize: '14px' }}>{input.title}</h3>
            <textarea
              value={input.targetText}
              readOnly
              className="neon-input"
              rows={Math.max(10, input.targetText.split('\n').length)}
              cols={50}
              style={{ resize: 'none', fontSize: '12px' }}
            />
          </div>
        ))}
      </div>
       
      <div className="core-module">
        <SendRequestToBackend />
        <MarkdownEditor value={markdownContent} />
      </div>

      <div className="tech-panel right-panel">
        {rightInputs.map((input, index) => (
          <div key={`right-${index}`} className="input-wrapper">
            <h3 className="input-title" style={{ fontSize: '14px' }}>{input.title}</h3>
            <textarea
              value={input.targetText}
              readOnly
              className="neon-input"
              rows={100}
              cols={50}
              style={{ resize: 'none', fontSize: '12px' }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;