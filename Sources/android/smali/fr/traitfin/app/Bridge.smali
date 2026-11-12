.class public Lfr/traitfin/app/Bridge;
.super Ljava/lang/Object;
.implements Ljava/lang/Runnable;

.field public a:Lfr/traitfin/app/MainActivity;
.field public name:Ljava/lang/String;

.method public constructor <init>(Lfr/traitfin/app/MainActivity;)V
    .registers 2
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V
    iput-object p1, p0, Lfr/traitfin/app/Bridge;->a:Lfr/traitfin/app/MainActivity;
    return-void
.end method

.method public saveSvg(Ljava/lang/String;Ljava/lang/String;)V
    .registers 4
    .annotation runtime Landroid/webkit/JavascriptInterface;
    .end annotation
    iget-object v0, p0, Lfr/traitfin/app/Bridge;->a:Lfr/traitfin/app/MainActivity;
    iput-object p2, v0, Lfr/traitfin/app/MainActivity;->pending:Ljava/lang/String;
    iput-object p1, p0, Lfr/traitfin/app/Bridge;->name:Ljava/lang/String;
    invoke-virtual {v0, p0}, Lfr/traitfin/app/MainActivity;->runOnUiThread(Ljava/lang/Runnable;)V
    return-void
.end method

.method public run()V
    .registers 5
    new-instance v0, Landroid/content/Intent;
    const-string v1, "android.intent.action.CREATE_DOCUMENT"
    invoke-direct {v0, v1}, Landroid/content/Intent;-><init>(Ljava/lang/String;)V
    const-string v1, "android.intent.category.OPENABLE"
    invoke-virtual {v0, v1}, Landroid/content/Intent;->addCategory(Ljava/lang/String;)Landroid/content/Intent;
    const-string v1, "image/svg+xml"
    invoke-virtual {v0, v1}, Landroid/content/Intent;->setType(Ljava/lang/String;)Landroid/content/Intent;
    const-string v1, "android.intent.extra.TITLE"
    iget-object v2, p0, Lfr/traitfin/app/Bridge;->name:Ljava/lang/String;
    invoke-virtual {v0, v1, v2}, Landroid/content/Intent;->putExtra(Ljava/lang/String;Ljava/lang/String;)Landroid/content/Intent;
    iget-object v1, p0, Lfr/traitfin/app/Bridge;->a:Lfr/traitfin/app/MainActivity;
    const/4 v2, 0x2
    :try_start
    invoke-virtual {v1, v0, v2}, Lfr/traitfin/app/MainActivity;->startActivityForResult(Landroid/content/Intent;I)V
    :try_end
    .catch Ljava/lang/Exception; {:try_start .. :try_end} :fail
    :fail
    return-void
.end method

.method public copy(Ljava/lang/String;)Z
    .registers 5
    .annotation runtime Landroid/webkit/JavascriptInterface;
    .end annotation
    :try_start
    iget-object v0, p0, Lfr/traitfin/app/Bridge;->a:Lfr/traitfin/app/MainActivity;
    const-string v1, "clipboard"
    invoke-virtual {v0, v1}, Landroid/app/Activity;->getSystemService(Ljava/lang/String;)Ljava/lang/Object;
    move-result-object v0
    check-cast v0, Landroid/content/ClipboardManager;
    const-string v1, "SVG"
    invoke-static {v1, p1}, Landroid/content/ClipData;->newPlainText(Ljava/lang/CharSequence;Ljava/lang/CharSequence;)Landroid/content/ClipData;
    move-result-object v1
    invoke-virtual {v0, v1}, Landroid/content/ClipboardManager;->setPrimaryClip(Landroid/content/ClipData;)V
    :try_end
    .catch Ljava/lang/Exception; {:try_start .. :try_end} :fail
    const/4 v0, 0x1
    return v0
    :fail
    const/4 v0, 0x0
    return v0
.end method
