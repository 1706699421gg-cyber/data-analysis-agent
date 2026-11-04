from dotenv import load_dotenv
load_dotenv()
import os

pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
sk = os.getenv("LANGFUSE_SECRET_KEY", "")
host = os.getenv("LANGFUSE_HOST", "")

print("1) Public Key:", f"{pk[:12]}..." if pk else "(没读到)")
print("2) Secret Key:", f"{sk[:12]}..." if sk else "(没读到)")
print("3) LANGFUSE_HOST:", host or "(未设置，SDK默认用 https://cloud.langfuse.com)")

if not pk or not sk or "xxxx" in pk or "xxxx" in sk:
    print(">>> 结论A：.env 里是占位符或没读到，去填真实 key")
else:
    from langfuse import Langfuse
    try:
        lf = Langfuse()
        ok = lf.auth_check()   # 拿 key 向服务器做一次真实鉴权
        print("4) 服务端鉴权:", "通过 ✅" if ok else "失败 ❌")
    except Exception as e:
        print("4) 服务端鉴权: 失败 ❌ ->", e)
