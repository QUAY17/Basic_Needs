{{/*
Expand the name of the chart.
*/}}
{{- define "basic-needs-dashboard.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "basic-needs-dashboard.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "basic-needs-dashboard.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "basic-needs-dashboard.labels" -}}
helm.sh/chart: {{ include "basic-needs-dashboard.chart" . }}
{{ include "basic-needs-dashboard.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "basic-needs-dashboard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "basic-needs-dashboard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Instantiate ingress url
*/}}
{{- define "basic-needs-dashboard.ingress.url" -}}
{{- if .Values.ingress -}}
    {{- if .Values.ingress.generateHostName }}
        {{- printf "%s-%s.%s" .Release.Name .Release.Namespace .Values.ingress.defaultDomain -}}
    {{- else }}
        {{- if not .Values.ingress.hostname }}
            {{- fail "value for .Values.ingress.generateHostname is true, but .Values.ingress.hostname is empty" }}
        {{- else }}
            {{- .Values.ingress.hostname }}
        {{- end }}
    {{- end }}
{{- else -}}
    {{- fail "value for .Values.ingress non-existent" }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "basic-needs-dashboard.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "basic-needs-dashboard.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}