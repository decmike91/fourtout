.class public Lfr/traitfin/app/MainActivity;
.super Landroid/app/Activity;

.field public web:Landroid/webkit/WebView;
.field public fileCb:Landroid/webkit/ValueCallback;
.field public pending:Ljava/lang/String;

.method public constructor <init>()V
    .registers 1
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method

.method protected onCreate(Landroid/os/Bundle;)V
    .registers 7
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V
    new-instance v0, Landroid/webkit/WebView;
    invoke-direct {v0, p0}, Landroid/webkit/WebView;-><init>(Landroid/content/Context;)V
    iput-object v0, p0, Lfr/traitfin/app/MainActivity;->web:Landroid/webkit/WebView;
    invoke-virtual {v0}, Landroid/webkit/WebView;->getSettings()Landroid/webkit/WebSettings;
    move-result-object v1
    const/4 v2, 0x1
    invoke-virtual {v1, v2}, Landroid/webkit/WebSettings;->setJavaScriptEnabled(Z)V
    invoke-virtual {v1, v2}, Landroid/webkit/WebSettings;->setDomStorageEnabled(Z)V
    invoke-virtual {v1, v2}, Landroid/webkit/WebSettings;->setAllowFileAccess(Z)V
    new-instance v3, Lfr/traitfin/app/Chrome;
    invoke-direct {v3, p0}, Lfr/traitfin/app/Chrome;-><init>(Lfr/traitfin/app/MainActivity;)V
    invoke-virtual {v0, v3}, Landroid/webkit/WebView;->setWebChromeClient(Landroid/webkit/WebChromeClient;)V
    new-instance v3, Lfr/traitfin/app/Bridge;
    invoke-direct {v3, p0}, Lfr/traitfin/app/Bridge;-><init>(Lfr/traitfin/app/MainActivity;)V
    const-string v4, "Android"
    invoke-virtual {v0, v3, v4}, Landroid/webkit/WebView;->addJavascriptInterface(Ljava/lang/Object;Ljava/lang/String;)V
    const-string v4, "file:///android_asset/index.html"
    invoke-virtual {v0, v4}, Landroid/webkit/WebView;->loadUrl(Ljava/lang/String;)V
    invoke-virtual {p0, v0}, Lfr/traitfin/app/MainActivity;->setContentView(Landroid/view/View;)V
    return-void
.end method

.method protected onActivityResult(IILandroid/content/Intent;)V
    .registers 8
    const/4 v0, 0x1
    if-ne p1, v0, :save
    iget-object v1, p0, Lfr/traitfin/app/MainActivity;->fileCb:Landroid/webkit/ValueCallback;
    if-eqz v1, :end
    invoke-static {p2, p3}, Landroid/webkit/WebChromeClient$FileChooserParams;->parseResult(ILandroid/content/Intent;)[Landroid/net/Uri;
    move-result-object v2
    invoke-interface {v1, v2}, Landroid/webkit/ValueCallback;->onReceiveValue(Ljava/lang/Object;)V
    const/4 v2, 0x0
    iput-object v2, p0, Lfr/traitfin/app/MainActivity;->fileCb:Landroid/webkit/ValueCallback;
    goto :end

    :save
    const/4 v0, 0x2
    if-ne p1, v0, :end
    const/4 v0, -0x1
    if-ne p2, v0, :end
    if-eqz p3, :end
    iget-object v1, p0, Lfr/traitfin/app/MainActivity;->pending:Ljava/lang/String;
    if-eqz v1, :end

    :try_start
    invoke-virtual {p3}, Landroid/content/Intent;->getData()Landroid/net/Uri;
    move-result-object v2
    invoke-virtual {p0}, Lfr/traitfin/app/MainActivity;->getContentResolver()Landroid/content/ContentResolver;
    move-result-object v3
    invoke-virtual {v3, v2}, Landroid/content/ContentResolver;->openOutputStream(Landroid/net/Uri;)Ljava/io/OutputStream;
    move-result-object v3
    const-string v2, "UTF-8"
    invoke-virtual {v1, v2}, Ljava/lang/String;->getBytes(Ljava/lang/String;)[B
    move-result-object v2
    invoke-virtual {v3, v2}, Ljava/io/OutputStream;->write([B)V
    invoke-virtual {v3}, Ljava/io/OutputStream;->close()V
    :try_end
    .catch Ljava/lang/Exception; {:try_start .. :try_end} :err

    const-string v2, "SVG enregistré"

    :toast
    const/4 v0, 0x0
    invoke-static {p0, v2, v0}, Landroid/widget/Toast;->makeText(Landroid/content/Context;Ljava/lang/CharSequence;I)Landroid/widget/Toast;
    move-result-object v0
    invoke-virtual {v0}, Landroid/widget/Toast;->show()V

    :end
    return-void

    :err
    const-string v2, "Échec de l\'enregistrement"
    goto :toast
.end method
