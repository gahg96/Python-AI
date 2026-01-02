const API_BASE_URL = "http://localhost:3000";

const getHeaders = (isMultipart = false) => {
    const headers: any = {};
    if (!isMultipart) {
        headers["Content-Type"] = "application/json";
    }
    if (typeof window !== "undefined") {
        const token = localStorage.getItem("token");
        if (token) {
            headers["Authorization"] = `Bearer ${token}`;
        }
    }
    return headers;
};

const handleResponse = async (res: Response) => {
    if (res.status === 401) {
        if (typeof window !== "undefined" && !window.location.pathname.includes('/login')) {
            window.location.href = '/login';
        }
        throw new Error("Unauthorized");
    }
    if (!res.ok) {
        const errorBody = await res.text();
        console.error(`API Error:`, res.status, errorBody);
        throw new Error(`API Error: ${res.statusText} - ${errorBody}`);
    }
    return res.json();
};

export const api = {
    get: async (endpoint: string) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: getHeaders(),
        });
        return handleResponse(res);
    },
    post: async (endpoint: string, data: any) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "POST",
            headers: getHeaders(),
            body: JSON.stringify(data),
        });
        return handleResponse(res);
    },
    patch: async (endpoint: string, data: any) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "PATCH",
            headers: getHeaders(),
            body: JSON.stringify(data),
        });
        return handleResponse(res);
    },
    upload: async (endpoint: string, formData: FormData) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "POST",
            headers: getHeaders(true),
            body: formData,
        });
        return handleResponse(res);
    },
    delete: async (endpoint: string) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "DELETE",
            headers: getHeaders(),
        });
        return handleResponse(res);
    }
};
