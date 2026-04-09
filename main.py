# -*- coding: utf-8 -*-
"""应用入口"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=8100,
        reload=True
    )