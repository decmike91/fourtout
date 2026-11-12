.class public Lfr/traitfin/app/Chrome;
.super Landroid/webkit/WebChromeClient;

.field public a:Lfr/traitfin/app/MainActivity;

.method public constructor <init>(Lfr/traitfin/app/MainActivity;)V
    .registers 2
    invoke-direct {p0}, Landroid/webkit/WebChromeClient;-><init>()V
    iput-object p1, p0, Lfr/traitfin/app/Chrome;->a:Lfr/traitfin/app/MainActivity;
    return-void
.end method

.method public onShowFileChooser(Landroid/webkit/WebView;Landroid/webkit/ValueCallback;Landroid/webkit/WebChromeClient$FileChooserParams;)Z
    .registers 8
    iget-object v0, p0, Lfr/traitfin/app/Chrome;->a:Lfr/traitfin/app/MainActivity;
    iget-object v1, v0, Lfr/traitfin/app/MainActivity;->fileCb:Landroid/webkit/ValueCallback;
    if-eqz v1, :nocb
    const/4 v2, 0x0
    invoke-interface {v1, v2}, Landroid/webkit/ValueCallback;->onReceiveValue(Ljava/lang/Object;)V
    :nocb
    iput-object p2, v0, Lfr/traitfin/app/MainActivity;->fileCb:Landroid/webkit/ValueCallback;

    :try_start
    invoke-virtual {p3}, Landroid/webkit/WebChromeClient$FileChooserParams;->createIntent()Landroid/content/Intent;
    move-result-object v1
    const/4 v2, 0x1
    invoke-virtual {v0, v1, v2}, Lfr/traitfin/app/MainActivity;->startActivityForResult(Landroid/content/Intent;I)V
    :try_end
    .catch Ljava/lang/Exception; {:try_start .. :try_end} :fail

    const/4 v1, 0x1
    return v1

    :fail
    const/4 v1, 0x0
    iput-object v1, v0, Lfr/traitfin/app/MainActivity;->fileCb:Landroid/webkit/ValueCallback;
    return v1
.end method
