# Deploying Streamlit App on AWS EC2

This guide will walk you through deploying your Sentiment Analysis application on an AWS EC2 instance.

## Prerequisites

1.  An **AWS Account**.
2.  Your code pushed to a **GitHub repository** (or another git provider). This is the easiest way to get code onto the server.

---

## Step 1: Launch an EC2 Instance

1.  Log in to the **AWS Console** and search for **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: Give your instance a name (e.g., `Sentiment-App`).
4.  **AMI (OS)**: Select **Ubuntu** (Ubuntu Server 24.04 or 22.04 LTS are good choices).
5.  **Instance Type**: Select **t2.micro** (Free Tier eligible) or larger if needed.
6.  **Key Pair**: Create a new key pair (Download the `.pem` file) or use an existing one. **Keep this file safe!**
7.  **Network Settings**:
    *   Check "Allow SSH traffic from Anywhere" (or My IP for better security).
    *   Check "Allow HTTP traffic from the internet".
    *   Check "Allow HTTPS traffic from the internet".
8.  **Launch** the instance.

## Step 2: Configure Security Group (Open Port 8501)

Streamlit runs on port **8501** by default. We need to open this port in the AWS firewall.

1.  Go to your EC2 Dashboard -> **Instances**.
2.  Click on your new instance.
3.  Go to the **Security** tab and click the **Security Group** link (e.g., `sg-0123...`).
4.  Click **Edit inbound rules**.
5.  Click **Add rule**:
    *   **Type**: Custom TCP
    *   **Port range**: `8501`
    *   **Source**: `0.0.0.0/0` (Anywhere IPv4)
6.  Click **Save rules**.

---

## Step 3: Connect to your Instance

1.  Open your terminal (or Command Prompt/PowerShell on Windows).
2.  Navigate to the folder where you saved your `.pem` key file.
3.  Run the following command (replace `your-key.pem` and `your-public-ip`):

```bash
# Set permissions (Linux/Mac only)
chmod 400 your-key.pem

# Connect
ssh -i "your-key.pem" ubuntu@your-public-ip
```

*(If you are on Windows and this looks difficult, you can use **EC2 Instance Connect** directly from the AWS Console browser).*

---

## Step 4: Setup the Environment

Once connected to the server, run these commands one by one to install Python and necessary tools:

```bash
# 1. Update system packages
sudo apt update

# 2. Install Python pip and venv


```

---

## Step 5: Clone Your Code

```bash
# 1. Clone your repository (Replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 2. Navigate into the folder
cd YOUR_REPO_NAME
```

*(Note: If you haven't pushed your code to GitHub yet, you can also use `scp` to upload files, but git is much easier for updates).*

---

## Step 6: Install Dependencies

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

---

## Step 7: Run the Application

Test if the application works:

```bash
streamlit run app.py
```

If it runs successfully, you can access your app at:
`http://your-ec2-public-ip:8501`

---

## Step 8: Keep App Running in Background

When you close the SSH terminal, the app will stop. To keep it running, use `tmux` or `nohup`. Here is the `tmux` method (recommended):

1.  **Stop** the current app (Ctrl+C).
2.  Install tmux:
    ```bash
    sudo apt install tmux -y
    ```
3.  Start a new session:
    ```bash
    tmux new -s sentiment_app
    ```
4.  Inside the new session, run the app:
    ```bash
    streamlit run app.py
    ```
5.  **Detach** from the session by pressing `Ctrl+B`, then `D`.
    *   Your app will keep running even if you disconnect.
    *   To attach back later: `tmux attach -t sentiment_app`

## Optional: Custom Domain / Port 80

If you want to access the app without `:8501` (i.e., just `http://your-ip`), you can run Streamlit on port 80:

```bash
sudo streamlit run app.py --server.port 80
```
*(Note: Running on port 80 usually requires root privileges or port forwarding).*
